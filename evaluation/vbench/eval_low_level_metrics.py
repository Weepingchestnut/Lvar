import argparse
import cv2
import json
import os
import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from utils.misc import time_str

try:
    from evaluation.vbench import temporal_metrics_utils, video_io_utils
except ImportError:  # running as a plain script: evaluation/vbench is sys.path[0]
    import temporal_metrics_utils
    import video_io_utils


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class VideoPair:
    key: str
    baseline_path: str
    candidate_path: str


@dataclass
class DimensionVideoPair:
    pair: VideoPair
    dimensions: List[str]


class RunningMean:
    def __init__(self) -> None:
        self.value_sum = 0.0
        self.count = 0

    def update(self, value_sum: float, count: int) -> None:
        self.value_sum += float(value_sum)
        self.count += int(count)

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.value_sum / self.count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate low-level video metrics between baseline and accelerated generations. "
        "Supports the VBench result layout and plain flat video directories (e.g. latency-profile outputs)."
    )
    parser.add_argument("--baseline-root", type=str, required=True, help="Baseline result root (VBench layout or flat video dir).")
    parser.add_argument("--candidate-root", type=str, required=True, help="Accelerated-model result root (VBench layout or flat video dir).")
    parser.add_argument(
        "--input-layout",
        choices=("auto", "vbench", "flat"),
        default="auto",
        help="'vbench' expects the VBench structure (videos/ + videos_by_dimension/). 'flat' treats "
        "both roots as plain directories of videos matched by filename (a root/videos subdir is also "
        "accepted); --dimensions is ignored in flat mode. 'auto' detects the layout from each root.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Where to save the summary json. Default: <candidate-root>/evaluation_results/low_level_metrics_vs_baseline.json",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=None,
        help="Optional dimension subset. Default: evaluate all shared dimensions.",
    )
    parser.add_argument(
        "--decode-backend",
        choices=("auto", "torchcodec", "decord", "opencv"),
        default="auto",
        help="Video decoding backend. 'auto' prefers torchcodec, then decord, then opencv. "
        "When the resolved backend is opencv, PNG frame directories are also read with the "
        "legacy cv2 reader (exact legacy-pipeline reproduction).",
    )
    parser.add_argument(
        "--decode-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="torchcodec only. 'cuda' decodes via NVDEC on the metric GPU; 'auto' tries NVDEC and "
        "permanently falls back to CPU decoding on the first failure.",
    )
    parser.add_argument(
        "--decode-threads",
        type=int,
        default=0,
        help="FFmpeg thread count per torchcodec decoder. 0 keeps FFmpeg's default.",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=2,
        help="How many video pairs to decode ahead in background threads (overlaps decoding with "
        "GPU metric computation). 0 disables prefetching.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Metric device, e.g. cuda, cuda:0, cpu. In distributed mode local rank will override cuda index.",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=16,
        help="How many frames to evaluate per GPU batch.",
    )
    parser.add_argument(
        "--lpips-net-type",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
        help="Backbone used by torchmetrics LPIPS.",
    )
    parser.add_argument(
        "--preferred-source",
        choices=("auto", "npy", "png", "video"),
        default="auto",
        help="Metric input source. 'auto' prefers npy, then PNG frame directories, then encoded videos.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=-1,
        help="Debug option. If > 0, only evaluate the first N matched pairs for overall and for each dimension.",
    )
    parser.add_argument(
        "--allow-frame-count-mismatch",
        action="store_true",
        help="If set, compare only the min(T_baseline, T_candidate) frames when frame counts differ.",
    )
    parser.add_argument(
        "--allow-spatial-mismatch",
        action="store_true",
        help="If set, resize candidate frames to baseline spatial size before metric computation.",
    )
    parser.add_argument(
        "--include-first-frame",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether to include frame 0 in metric computation. Set to 0 to evaluate only later video frames.",
    )
    parser.add_argument(
        "--collect-per-video",
        action="store_true",
        help="Gather per-video metric values to rank 0; adds a per_video_metrics section and std "
        "fields to the report json.",
    )
    parser.add_argument(
        "--compute-fvd",
        action="store_true",
        help="Also compute FVD between candidate and baseline videos (overall only) using "
        "styleganv I3D features.",
    )
    parser.add_argument(
        "--i3d-path",
        type=str,
        default=None,
        help="Path to styleganv `i3d_torchscript.pt` (required with --compute-fvd).",
    )
    parser.add_argument(
        "--compute-warp-error",
        action="store_true",
        help="Also compute RAFT flow warping error (per-video temporal consistency) for baseline "
        "and candidate plus their delta.",
    )
    parser.add_argument(
        "--raft-model",
        choices=("small", "large"),
        default="small",
        help="RAFT variant for --compute-warp-error. 'large' is more accurate but much slower.",
    )
    parser.add_argument(
        "--warp-chunk-size",
        type=int,
        default=2,
        help="Frame pairs per RAFT forward pass (VRAM knob for --compute-warp-error).",
    )
    parser.add_argument(
        "--warp-downscale",
        type=float,
        default=1.0,
        help="Spatial downscale applied before flow computation, e.g. 0.5 halves H/W (the warping "
        "error is then measured at that scale).",
    )
    parser.add_argument(
        "--warp-frame-stride",
        type=int,
        default=1,
        help="Stride between the two frames of each warping pair (t vs t+stride).",
    )
    return parser.parse_args()


def maybe_init_distributed(device_arg: str) -> Tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = 0

    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        if device_arg.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(device_arg)
        else:
            device = torch.device("cpu")

    return device, rank, world_size, distributed


def resolve_vbench_dirs(root: str) -> Tuple[str, str, str, str]:
    root = osp.abspath(root)
    basename = osp.basename(root)
    if basename == "videos":
        result_root = osp.dirname(root)
        videos_dir = root
        dims_dir = osp.join(result_root, "videos_by_dimension")
        frames_dims_dir = osp.join(result_root, "frames_by_dimension")
    elif basename == "videos_by_dimension":
        result_root = osp.dirname(root)
        videos_dir = osp.join(result_root, "videos")
        dims_dir = root
        frames_dims_dir = osp.join(result_root, "frames_by_dimension")
    elif basename == "frames_by_dimension":
        result_root = osp.dirname(root)
        videos_dir = osp.join(result_root, "videos")
        dims_dir = osp.join(result_root, "videos_by_dimension")
        frames_dims_dir = root
    else:
        result_root = root
        videos_dir = osp.join(root, "videos")
        dims_dir = osp.join(root, "videos_by_dimension")
        frames_dims_dir = osp.join(root, "frames_by_dimension")

    if not osp.isdir(videos_dir):
        raise FileNotFoundError(f"Cannot find videos directory under: {root}")
    if not osp.isdir(dims_dir):
        raise FileNotFoundError(f"Cannot find videos_by_dimension directory under: {root}")

    return result_root, videos_dir, dims_dir, frames_dims_dir


def list_media_files(directory: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for name in sorted(os.listdir(directory)):
        path = osp.join(directory, name)
        if not osp.isfile(path) and not osp.islink(path):
            continue
        suffix = osp.splitext(name)[1].lower()
        if suffix not in VIDEO_EXTENSIONS:
            continue
        files[name] = path
    return files


def list_video_files(directory: str) -> Dict[str, str]:
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return {
        name: path
        for name, path in list_media_files(directory).items()
        if osp.splitext(name)[1].lower() in video_suffixes
    }


def resolve_flat_videos_dir(root: str) -> Optional[str]:
    """Return the directory actually holding the videos for flat-layout roots.

    Accepts either a directory that directly contains video files, or a root
    whose `videos/` subdirectory contains them (latency-profile convention).
    """
    root = osp.abspath(root)
    if osp.isdir(root) and list_video_files(root):
        return root
    nested = osp.join(root, "videos")
    if osp.isdir(nested) and list_video_files(nested):
        return nested
    return None


def detect_input_layout(root: str) -> str:
    try:
        resolve_vbench_dirs(root)
        return "vbench"
    except FileNotFoundError:
        pass
    if resolve_flat_videos_dir(root) is not None:
        return "flat"
    raise FileNotFoundError(
        f"Cannot interpret {root} as a VBench result root (videos/ + videos_by_dimension/) "
        "or as a flat video directory."
    )


def resolve_real_path(path: str) -> str:
    return osp.realpath(path) if osp.islink(path) else path


def select_metric_source(
    video_path: str,
    preferred_source: str,
    frame_dir: Optional[str] = None,
) -> str:
    real_video_path = resolve_real_path(video_path)
    video_stem, _ = osp.splitext(real_video_path)
    npy_path = f"{video_stem}.npy"
    physical_frame_dir = video_stem
    dim_npy_path = f"{frame_dir}.npy" if frame_dir is not None else None

    candidates: List[str] = []
    if preferred_source == "auto":
        if dim_npy_path is not None:
            candidates.append(dim_npy_path)
        candidates.append(npy_path)
        if frame_dir is not None:
            candidates.append(frame_dir)
        candidates.extend([physical_frame_dir, video_path])
    elif preferred_source == "npy":
        if dim_npy_path is not None:
            candidates.append(dim_npy_path)
        candidates.append(npy_path)
    elif preferred_source == "png":
        if frame_dir is not None:
            candidates.append(frame_dir)
        candidates.append(physical_frame_dir)
    elif preferred_source == "video":
        candidates = [video_path]

    for candidate in candidates:
        if candidate and osp.exists(candidate):
            return candidate

    if preferred_source == "auto":
        return video_path
    raise FileNotFoundError(f"Cannot find requested {preferred_source} source for: {video_path}")


def collect_overall_pairs(
    baseline_videos_dir: str,
    candidate_videos_dir: str,
    limit_videos: int,
    preferred_source: str,
) -> Tuple[List[VideoPair], Dict[str, int]]:
    baseline_files = list_video_files(baseline_videos_dir)
    candidate_files = list_video_files(candidate_videos_dir)

    shared_keys = sorted(set(baseline_files) & set(candidate_files))
    if limit_videos > 0:
        shared_keys = shared_keys[:limit_videos]

    pairs = [
        VideoPair(
            key=key,
            baseline_path=select_metric_source(baseline_files[key], preferred_source),
            candidate_path=select_metric_source(candidate_files[key], preferred_source),
        )
        for key in shared_keys
    ]
    stats = {
        "baseline_total": len(baseline_files),
        "candidate_total": len(candidate_files),
        "shared_total": len(shared_keys),
        "baseline_only": len(set(baseline_files) - set(candidate_files)),
        "candidate_only": len(set(candidate_files) - set(baseline_files)),
    }
    return pairs, stats


def collect_dimension_pairs(
    baseline_dims_dir: str,
    candidate_dims_dir: str,
    baseline_frames_dims_dir: str,
    candidate_frames_dims_dir: str,
    selected_dimensions: Optional[Sequence[str]],
    preferred_source: str,
) -> Tuple[Dict[str, List[VideoPair]], Dict[str, Dict[str, int]]]:
    baseline_dims = {name for name in os.listdir(baseline_dims_dir) if osp.isdir(osp.join(baseline_dims_dir, name))}
    candidate_dims = {name for name in os.listdir(candidate_dims_dir) if osp.isdir(osp.join(candidate_dims_dir, name))}
    shared_dims = sorted(baseline_dims & candidate_dims)
    if selected_dimensions:
        selected_set = set(selected_dimensions)
        shared_dims = [dim for dim in shared_dims if dim in selected_set]

    pairs_by_dim: Dict[str, List[VideoPair]] = {}
    stats_by_dim: Dict[str, Dict[str, int]] = {}

    for dim in shared_dims:
        baseline_files = list_video_files(osp.join(baseline_dims_dir, dim))
        candidate_files = list_video_files(osp.join(candidate_dims_dir, dim))
        shared_keys = sorted(set(baseline_files) & set(candidate_files))
        pairs_by_dim[dim] = []
        for key in shared_keys:
            key_stem, _ = osp.splitext(key)
            baseline_frame_dir = osp.join(baseline_frames_dims_dir, dim, key_stem)
            candidate_frame_dir = osp.join(candidate_frames_dims_dir, dim, key_stem)
            pairs_by_dim[dim].append(
                VideoPair(
                    key=key,
                    baseline_path=select_metric_source(
                        baseline_files[key],
                        preferred_source,
                        frame_dir=baseline_frame_dir,
                    ),
                    candidate_path=select_metric_source(
                        candidate_files[key],
                        preferred_source,
                        frame_dir=candidate_frame_dir,
                    ),
                )
            )
        stats_by_dim[dim] = {
            "baseline_total": len(baseline_files),
            "candidate_total": len(candidate_files),
            "shared_total": len(shared_keys),
            "baseline_only": len(set(baseline_files) - set(candidate_files)),
            "candidate_only": len(set(candidate_files) - set(baseline_files)),
        }

    return pairs_by_dim, stats_by_dim


def choose_decoder_backend(preferred: str) -> str:
    if preferred in ("torchcodec", "auto"):
        try:
            import torchcodec  # noqa: F401

            return "torchcodec"
        except ImportError:
            if preferred == "torchcodec":
                raise
    if preferred in ("decord", "auto"):
        try:
            # import cv2     # import opencv first for bug: libpng error: bad parameters to zlib
            import decord  # noqa: F401

            return "decord"
        except ImportError:
            if preferred == "decord":
                raise
    try:
        import cv2  # noqa: F401

        return "opencv"
    except ImportError as exc:
        raise ImportError(
            "None of torchcodec / decord / opencv-python is available for video decoding."
        ) from exc


def read_frame_dir_rgb(frame_dir: str, backend: str = "auto") -> torch.Tensor:
    # [T, C, H, W] RGB uint8. An explicitly selected opencv backend also forces
    # the legacy cv2 frame reader (exact legacy-pipeline reproduction);
    # otherwise torchvision.io + thread pool (cv2 fallback inside).
    return video_io_utils.read_frame_dir_rgb(frame_dir, force_cv2=(backend == "opencv"))


def read_video_rgb(video_path: str, backend: str, decoder=None) -> torch.Tensor:
    """Decode any supported source into a [T, C, H, W] RGB uint8 tensor.

    `decoder` is a video_io_utils.TorchcodecDecoder, only used when
    backend == "torchcodec"; its frames may already live on GPU (NVDEC).
    """
    if osp.isdir(video_path):
        return read_frame_dir_rgb(video_path, backend)

    suffix = osp.splitext(video_path)[1].lower()
    if suffix == ".npy":
        frames = np.load(video_path)
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim != 4:
            raise RuntimeError(f"Expected npy frames with shape [T,H,W,3], got {frames.shape}: {video_path}")
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

    if suffix in IMAGE_EXTENSIONS:
        # No local `import cv2` here: a function-local import would make `cv2`
        # local to this whole function and break the opencv video branch below
        # (UnboundLocalError). The module-level import always provides it.
        image = cv2.imread(video_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {video_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()

    if backend == "torchcodec":
        if decoder is None:
            decoder = video_io_utils.TorchcodecDecoder("cpu", torch.device("cpu"))
        return decoder.decode(video_path)

    if backend == "decord":
        # import cv2
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frame_ids = list(range(len(vr)))
        frames = vr.get_batch(frame_ids).asnumpy()
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

    # import cv2

    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Decoded zero frames from: {video_path}")
    return torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).contiguous()


def build_metric_modules(device: torch.device, lpips_net_type: str):
    try:
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    except ImportError:
        try:
            from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        except ImportError as exc:
            raise ImportError(
                "torchmetrics is required. Please install torchmetrics and lpips in the evaluation environment."
            ) from exc
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required. Please install torchmetrics and lpips in the evaluation environment."
        ) from exc

    psnr_metric = PeakSignalNoiseRatio(
        data_range=1.0,
        reduction="sum",
        dim=(1, 2, 3),
    ).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        reduction="sum",
    ).to(device)
    try:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type=lpips_net_type,
            reduction="sum",
            normalize=True,
        ).to(device)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "LPIPS metric requires the optional lpips package. Please install torchmetrics[image] or lpips."
        ) from exc
    lpips_metric.eval()
    return psnr_metric, ssim_metric, lpips_metric


def move_video_to_device(video: torch.Tensor, device: torch.device) -> torch.Tensor:
    # video: [T, C, H, W] RGB storing 0-255 (uint8, or float for npy sources).
    # Transfer before normalizing: uint8 moves 4x less PCIe data than float32.
    if video.device != device:
        video = video.to(device, non_blocking=True)
    return video.float().div_(255.0)


def maybe_resize_candidate(candidate: torch.Tensor, baseline_hw: Tuple[int, int]) -> torch.Tensor:
    if tuple(candidate.shape[-2:]) == baseline_hw:
        return candidate
    return torch.nn.functional.interpolate(
        candidate,
        size=baseline_hw,
        mode="bilinear",
        align_corners=False,
    )


def _locate_identical_frames(candidate: torch.Tensor, baseline: torch.Tensor, offset: int) -> List[int]:
    # Frames with MSE exactly 0 are the ones driving per-frame PSNR to inf.
    mse = (candidate - baseline).pow(2).flatten(start_dim=1).mean(dim=1)
    return [offset + i for i in torch.nonzero(mse == 0).flatten().tolist()]


def evaluate_video_pair(
    pair: VideoPair,
    decoder_backend: str,
    device: torch.device,
    frame_batch_size: int,
    allow_frame_count_mismatch: bool,
    allow_spatial_mismatch: bool,
    include_first_frame: bool,
    metrics,
    decoder=None,
    baseline_frames: Optional[torch.Tensor] = None,
    candidate_frames: Optional[torch.Tensor] = None,
    temporal=None,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Return per-pair averaged PSNR/SSIM/LPIPS (and optional temporal metrics).

    `baseline_frames` / `candidate_frames` accept pre-decoded [T, C, H, W] RGB
    uint8 tensors from the prefetcher; when omitted the videos are decoded here.
    `temporal` is an optional temporal_metrics_utils.TemporalMetricModules:
    warping errors are added as regular metric keys, I3D features (for FVD) are
    returned under the private key "_fvd_feats" and popped by the caller.
    """
    psnr_metric, ssim_metric, lpips_metric = metrics

    if baseline_frames is None:
        baseline_frames = read_video_rgb(pair.baseline_path, decoder_backend, decoder)   # [T, C, H, W] uint8
    if candidate_frames is None:
        candidate_frames = read_video_rgb(pair.candidate_path, decoder_backend, decoder)

    if baseline_frames.ndim != 4 or candidate_frames.ndim != 4:
        return None, f"decoded tensor rank mismatch for {pair.key}"

    if not include_first_frame:
        if baseline_frames.shape[0] <= 1 or candidate_frames.shape[0] <= 1:
            return None, f"cannot exclude first frame for {pair.key}: not enough frames"
        baseline_frames = baseline_frames[1:]
        candidate_frames = candidate_frames[1:]

    baseline_t = baseline_frames.shape[0]
    candidate_t = candidate_frames.shape[0]
    baseline_h, baseline_w = baseline_frames.shape[-2:]
    candidate_h, candidate_w = candidate_frames.shape[-2:]

    if (baseline_h, baseline_w) != (candidate_h, candidate_w) and not allow_spatial_mismatch:
        return None, (
            f"spatial mismatch for {pair.key}: "
            f"baseline={baseline_h}x{baseline_w}, candidate={candidate_h}x{candidate_w}"
        )

    if baseline_t != candidate_t and not allow_frame_count_mismatch:
        return None, f"frame count mismatch for {pair.key}: baseline={baseline_t}, candidate={candidate_t}"

    compare_t = min(baseline_t, candidate_t)
    if compare_t <= 0:
        return None, f"zero comparable frames for {pair.key}"
    baseline_frames = baseline_frames[:compare_t]
    candidate_frames = candidate_frames[:compare_t]

    baseline_frames = move_video_to_device(baseline_frames, device)         # [T, C, H, W] float in [0, 1]
    candidate_frames = move_video_to_device(candidate_frames, device)
    if allow_spatial_mismatch:
        candidate_frames = maybe_resize_candidate(candidate_frames, tuple(baseline_frames.shape[-2:]))

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0

    with torch.inference_mode():
        for start in range(0, compare_t, frame_batch_size):
            end = min(start + frame_batch_size, compare_t)
            baseline_chunk = baseline_frames[start:end]
            candidate_chunk = candidate_frames[start:end]

            psnr_chunk = psnr_metric(candidate_chunk, baseline_chunk)
            psnr_metric.reset()
            if not bool(torch.isfinite(psnr_chunk)):
                bad_frames = _locate_identical_frames(candidate_chunk, baseline_chunk, start)
                return None, (
                    f"PSNR is non-finite for {pair.key} at frame indices {bad_frames or 'unknown'} "
                    "(candidate pixel-identical to baseline => MSE=0); pair excluded from averages"
                )
            psnr_sum += float(psnr_chunk.item())

            ssim_sum += float(ssim_metric(candidate_chunk, baseline_chunk).item())
            ssim_metric.reset()

            lpips_sum += float(lpips_metric(candidate_chunk, baseline_chunk).item())
            lpips_metric.reset()

    metrics_dict = {
        "psnr": psnr_sum / compare_t,
        "ssim": ssim_sum / compare_t,
        "lpips": lpips_sum / compare_t,
        "num_frames": compare_t,
    }

    # Temporal metrics reuse the exact frames the low-level metrics ran on
    # (same first-frame exclusion, length alignment, and candidate resize).
    if temporal is not None and temporal.warp_enabled:
        warp_baseline = temporal_metrics_utils.compute_warping_error(
            baseline_frames,
            temporal.raft,
            chunk_size=temporal.warp_chunk_size,
            downscale=temporal.warp_downscale,
            frame_stride=temporal.warp_frame_stride,
        )
        warp_candidate = temporal_metrics_utils.compute_warping_error(
            candidate_frames,
            temporal.raft,
            chunk_size=temporal.warp_chunk_size,
            downscale=temporal.warp_downscale,
            frame_stride=temporal.warp_frame_stride,
        )
        if warp_baseline is None or warp_candidate is None:
            return None, f"cannot compute warping error for {pair.key}: not enough frames"
        metrics_dict["warp_err_baseline"] = warp_baseline
        metrics_dict["warp_err_candidate"] = warp_candidate
        metrics_dict["warp_err_delta"] = warp_candidate - warp_baseline

    if temporal is not None and temporal.fvd_enabled:
        feat_baseline = temporal_metrics_utils.extract_i3d_features(baseline_frames, temporal.i3d)
        feat_candidate = temporal_metrics_utils.extract_i3d_features(candidate_frames, temporal.i3d)
        if feat_baseline is not None and feat_candidate is not None:
            metrics_dict["_fvd_feats"] = (feat_baseline, feat_candidate)

    return metrics_dict, None


def shard_list(items: Sequence[VideoPair], rank: int, world_size: int) -> List[VideoPair]:
    return list(items[rank::world_size])


def all_reduce_float(value: float, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_int(value: int, device: torch.device, distributed: bool) -> int:
    tensor = torch.tensor([value], dtype=torch.int64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def evaluate_scope(
    pairs: Sequence[VideoPair],
    decoder_backend: str,
    device: torch.device,
    frame_batch_size: int,
    allow_frame_count_mismatch: bool,
    allow_spatial_mismatch: bool,
    include_first_frame: bool,
    metrics,
    rank: int,
    world_size: int,
    distributed: bool,
    scope_name: str,
) -> Dict[str, Optional[float]]:
    local_pairs = shard_list(pairs, rank, world_size)
    psnr_meter = RunningMean()
    ssim_meter = RunningMean()
    lpips_meter = RunningMean()
    local_errors = 0

    for local_idx, pair in enumerate(local_pairs):
        result, error = evaluate_video_pair(
            pair=pair,
            decoder_backend=decoder_backend,
            device=device,
            frame_batch_size=frame_batch_size,
            allow_frame_count_mismatch=allow_frame_count_mismatch,
            allow_spatial_mismatch=allow_spatial_mismatch,
            include_first_frame=include_first_frame,
            metrics=metrics,
        )
        if error is not None:
            local_errors += 1
            print(f"[Rank {rank}] Skip {scope_name} pair {pair.key}: {error}")
            continue

        assert result is not None
        psnr_meter.update(result["psnr"], 1)
        ssim_meter.update(result["ssim"], 1)
        lpips_meter.update(result["lpips"], 1)

        if local_idx % 50 == 0:
            print(
                f"[Rank {rank}] {scope_name}: processed {local_idx + 1}/{len(local_pairs)} local pairs",
                flush=True,
            )

    global_psnr_sum = all_reduce_float(psnr_meter.value_sum, device, distributed)
    global_ssim_sum = all_reduce_float(ssim_meter.value_sum, device, distributed)
    global_lpips_sum = all_reduce_float(lpips_meter.value_sum, device, distributed)
    global_count = all_reduce_int(psnr_meter.count, device, distributed)
    global_errors = all_reduce_int(local_errors, device, distributed)

    if global_count == 0:
        return {
            "num_pairs": 0,
            "num_failed_pairs": global_errors,
            "psnr": None,
            "ssim": None,
            "lpips": None,
        }

    return {
        "num_pairs": global_count,
        "num_failed_pairs": global_errors,
        "psnr": global_psnr_sum / global_count,
        "ssim": global_ssim_sum / global_count,
        "lpips": global_lpips_sum / global_count,
    }


def pair_identity(pair: VideoPair) -> Tuple[str, str]:
    return (osp.realpath(pair.baseline_path), osp.realpath(pair.candidate_path))


def collect_unique_dimension_pairs(dim_pairs: Dict[str, List[VideoPair]]) -> List[DimensionVideoPair]:
    unique_items: Dict[Tuple[str, str], DimensionVideoPair] = {}
    for dim in sorted(dim_pairs):
        for pair in dim_pairs[dim]:
            identity = pair_identity(pair)
            if identity not in unique_items:
                unique_items[identity] = DimensionVideoPair(pair=pair, dimensions=[])
            if dim not in unique_items[identity].dimensions:
                unique_items[identity].dimensions.append(dim)
    return list(unique_items.values())


LOW_LEVEL_METRIC_KEYS = ["psnr", "ssim", "lpips"]
WARP_METRIC_KEYS = ["warp_err_baseline", "warp_err_candidate", "warp_err_delta"]


def make_meter_set(metric_keys: Sequence[str]) -> Dict[str, RunningMean]:
    return {key: RunningMean() for key in metric_keys}


def update_meter_set(meters: Dict[str, RunningMean], result: Dict[str, float]) -> None:
    for key, meter in meters.items():
        meter.update(result[key], 1)


def reduce_meter_set(
    meters: Dict[str, RunningMean],
    local_errors: int,
    device: torch.device,
    distributed: bool,
) -> Dict[str, Optional[float]]:
    metric_sums = {
        key: all_reduce_float(meter.value_sum, device, distributed) for key, meter in meters.items()
    }
    global_count = all_reduce_int(next(iter(meters.values())).count, device, distributed)
    global_errors = all_reduce_int(local_errors, device, distributed)

    summary: Dict[str, Optional[float]] = {
        "num_pairs": global_count,
        "num_failed_pairs": global_errors,
    }
    for key in meters:
        summary[key] = (metric_sums[key] / global_count) if global_count > 0 else None
    return summary


def gather_object_lists(local_list: List[Any], world_size: int, distributed: bool) -> List[Any]:
    if not distributed:
        return list(local_list)
    gathered: List[Optional[List[Any]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, list(local_list))
    merged: List[Any] = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def attach_std_fields(
    overall_result: Dict[str, Optional[float]],
    per_dimension_results: Dict[str, Dict[str, Optional[float]]],
    per_video_records: List[Dict[str, Any]],
    metric_keys: Sequence[str],
) -> None:
    # Std across per-video means; mirrors the mean aggregation (rank 0 only).
    def _stds(records: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        return {
            f"{key}_std": (float(np.std([rec[key] for rec in records])) if records else None)
            for key in metric_keys
        }

    overall_result.update(_stds(per_video_records))
    for dim, dim_result in per_dimension_results.items():
        dim_records = [rec for rec in per_video_records if dim in rec["dimensions"]]
        dim_result.update(_stds(dim_records))


def evaluate_unique_dimension_pairs(
    unique_items: Sequence[DimensionVideoPair],
    dim_names: Sequence[str],
    decoder_backend: str,
    device: torch.device,
    frame_batch_size: int,
    allow_frame_count_mismatch: bool,
    allow_spatial_mismatch: bool,
    include_first_frame: bool,
    metrics,
    rank: int,
    world_size: int,
    distributed: bool,
    metric_keys: Sequence[str] = tuple(LOW_LEVEL_METRIC_KEYS),
    decoder=None,
    prefetch_depth: int = 0,
    temporal=None,
    collect_per_video: bool = False,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Optional[float]]], Dict[str, Any]]:
    local_items = list(unique_items[rank::world_size])
    overall_meters = make_meter_set(metric_keys)
    dim_meters = {dim: make_meter_set(metric_keys) for dim in dim_names}
    local_overall_errors = 0
    local_dim_errors = {dim: 0 for dim in dim_names}
    local_per_video: List[Dict[str, Any]] = []
    local_failed: List[Dict[str, str]] = []
    local_fvd_baseline: List[np.ndarray] = []
    local_fvd_candidate: List[np.ndarray] = []
    local_fvd_skipped = 0

    def decode_pair(item: DimensionVideoPair):
        baseline = read_video_rgb(item.pair.baseline_path, decoder_backend, decoder)
        candidate = read_video_rgb(item.pair.candidate_path, decoder_backend, decoder)
        if device.type == "cuda":
            baseline = video_io_utils.maybe_pin_memory(baseline)
            candidate = video_io_utils.maybe_pin_memory(candidate)
        return baseline, candidate

    if prefetch_depth > 0:
        iterator = iter(video_io_utils.PairPrefetcher(local_items, decode_pair, depth=prefetch_depth))
    else:
        iterator = ((idx, item, None) for idx, item in enumerate(local_items))

    for local_idx, item, payload in iterator:
        if isinstance(payload, Exception):
            result, error = None, (
                f"{type(payload).__name__}: {payload}; "
                f"baseline_source={item.pair.baseline_path}; candidate_source={item.pair.candidate_path}"
            )
        else:
            baseline_frames, candidate_frames = payload if payload is not None else (None, None)
            try:
                result, error = evaluate_video_pair(
                    pair=item.pair,
                    decoder_backend=decoder_backend,
                    device=device,
                    frame_batch_size=frame_batch_size,
                    allow_frame_count_mismatch=allow_frame_count_mismatch,
                    allow_spatial_mismatch=allow_spatial_mismatch,
                    include_first_frame=include_first_frame,
                    metrics=metrics,
                    decoder=decoder,
                    baseline_frames=baseline_frames,
                    candidate_frames=candidate_frames,
                    temporal=temporal,
                )
            except Exception as exc:
                result, error = None, (
                    f"{type(exc).__name__}: {exc}; "
                    f"baseline_source={item.pair.baseline_path}; candidate_source={item.pair.candidate_path}"
                )

        if error is not None:
            local_overall_errors += 1
            for dim in item.dimensions:
                local_dim_errors[dim] += 1
            local_failed.append({"key": item.pair.key, "error": error})
            print(f"[Rank {rank}] [ERROR] Skip unique pair {item.pair.key}: {error}", flush=True)
            continue

        assert result is not None
        fvd_feats = result.pop("_fvd_feats", None)
        if temporal is not None and temporal.fvd_enabled:
            if fvd_feats is not None:
                local_fvd_baseline.append(fvd_feats[0])
                local_fvd_candidate.append(fvd_feats[1])
            else:
                local_fvd_skipped += 1

        update_meter_set(overall_meters, result)
        for dim in item.dimensions:
            update_meter_set(dim_meters[dim], result)
        if collect_per_video:
            record = {
                "key": item.pair.key,
                "dimensions": list(item.dimensions),
                "num_frames": result["num_frames"],
            }
            record.update({key: result[key] for key in metric_keys})
            local_per_video.append(record)

        if local_idx % 50 == 0:
            print(
                f"[Rank {rank}] unique dimension-union pairs: processed {local_idx + 1}/{len(local_items)} local pairs",
                flush=True,
            )

    overall_result = reduce_meter_set(overall_meters, local_overall_errors, device, distributed)
    per_dimension_results = {
        dim: reduce_meter_set(dim_meters[dim], local_dim_errors[dim], device, distributed)
        for dim in dim_names
    }
    extras = {
        "per_video": local_per_video,
        "failed": local_failed,
        "fvd_baseline": local_fvd_baseline,
        "fvd_candidate": local_fvd_candidate,
        "fvd_skipped": local_fvd_skipped,
    }
    return overall_result, per_dimension_results, extras


def default_output_json(candidate_root: str) -> str:
    save_dir = osp.join(candidate_root, "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)
    return osp.join(save_dir, f"low_level_metrics_vs_baseline_{time_str()}.json")


def main() -> None:
    args = parse_args()
    device, rank, world_size, distributed = maybe_init_distributed(args.device)

    layout = args.input_layout
    if layout == "auto":
        baseline_layout = detect_input_layout(args.baseline_root)
        candidate_layout = detect_input_layout(args.candidate_root)
        if baseline_layout != candidate_layout:
            raise ValueError(
                f"Input layout mismatch: baseline={baseline_layout}, candidate={candidate_layout}; "
                "pass --input-layout explicitly."
            )
        layout = baseline_layout

    if layout == "vbench":
        baseline_root, baseline_videos_dir, baseline_dims_dir, baseline_frames_dims_dir = resolve_vbench_dirs(args.baseline_root)
        candidate_root, candidate_videos_dir, candidate_dims_dir, candidate_frames_dims_dir = resolve_vbench_dirs(args.candidate_root)
        report_root = candidate_root
    else:
        baseline_videos_dir = resolve_flat_videos_dir(args.baseline_root)
        candidate_videos_dir = resolve_flat_videos_dir(args.candidate_root)
        if baseline_videos_dir is None:
            raise FileNotFoundError(f"No video files found under: {args.baseline_root}")
        if candidate_videos_dir is None:
            raise FileNotFoundError(f"No video files found under: {args.candidate_root}")
        baseline_root = osp.abspath(args.baseline_root)
        candidate_root = osp.abspath(args.candidate_root)
        # Keep evaluation_results/ next to (not inside) a videos/ directory.
        report_root = osp.dirname(candidate_videos_dir) if osp.basename(candidate_videos_dir) == "videos" else candidate_videos_dir
    output_json = osp.abspath(args.output_json) if args.output_json else default_output_json(report_root)

    decoder_backend = choose_decoder_backend(args.decode_backend)
    metrics = build_metric_modules(device=device, lpips_net_type=args.lpips_net_type)

    decoder = None
    if decoder_backend == "torchcodec":
        decoder = video_io_utils.TorchcodecDecoder(args.decode_device, device, args.decode_threads)
    temporal = temporal_metrics_utils.build_temporal_modules(
        compute_fvd=args.compute_fvd,
        i3d_path=args.i3d_path,
        compute_warp_error=args.compute_warp_error,
        raft_model=args.raft_model,
        warp_chunk_size=args.warp_chunk_size,
        warp_downscale=args.warp_downscale,
        warp_frame_stride=args.warp_frame_stride,
        device=device,
    )
    metric_keys = list(LOW_LEVEL_METRIC_KEYS)
    if temporal is not None and temporal.warp_enabled:
        metric_keys += WARP_METRIC_KEYS

    if layout == "vbench":
        dim_pairs, dim_pair_stats = collect_dimension_pairs(
            baseline_dims_dir=baseline_dims_dir,
            candidate_dims_dir=candidate_dims_dir,
            baseline_frames_dims_dir=baseline_frames_dims_dir,
            candidate_frames_dims_dir=candidate_frames_dims_dir,
            selected_dimensions=args.dimensions,
            preferred_source=args.preferred_source,
        )
        if args.limit_videos > 0:
            dim_pairs = {dim: pairs[: args.limit_videos] for dim, pairs in dim_pairs.items()}
            for dim, pairs in dim_pairs.items():
                dim_pair_stats[dim]["shared_total"] = len(pairs)

        unique_dimension_items = collect_unique_dimension_pairs(dim_pairs)
        dim_memberships_total = sum(len(pairs) for pairs in dim_pairs.values())
        baseline_video_files = list_video_files(baseline_videos_dir)
        candidate_video_files = list_video_files(candidate_videos_dir)
        overall_pair_stats = {
            "baseline_total": len(baseline_video_files),
            "candidate_total": len(candidate_video_files),
            "shared_total": len(unique_dimension_items),
            "dimension_memberships_total": dim_memberships_total,
            "baseline_only": len(set(baseline_video_files) - set(candidate_video_files)),
            "candidate_only": len(set(candidate_video_files) - set(baseline_video_files)),
            "aggregation": "unique union of videos referenced by selected dimensions",
        }
    else:
        flat_pairs, flat_stats = collect_overall_pairs(
            baseline_videos_dir=baseline_videos_dir,
            candidate_videos_dir=candidate_videos_dir,
            limit_videos=args.limit_videos,
            preferred_source=args.preferred_source,
        )
        dim_pairs = {}
        dim_pair_stats = {}
        unique_dimension_items = [DimensionVideoPair(pair=pair, dimensions=[]) for pair in flat_pairs]
        overall_pair_stats = {
            **flat_stats,
            "aggregation": "all videos matched by filename between the two flat directories",
        }

    if rank == 0:
        print("========== Low-Level Metric Evaluation ==========")
        print(f"Baseline root  : {baseline_root}")
        print(f"Candidate root : {candidate_root}")
        print(f"Input layout   : {layout}")
        print(f"Decoder backend: {decoder_backend}")
        if decoder is not None:
            print(f"Decode device  : {decoder.device_str} (torchcodec)")
        print(f"Prefetch depth : {args.prefetch_depth}")
        print(f"Temporal       : fvd={args.compute_fvd}, warp_error={args.compute_warp_error}")
        print(f"Preferred source: {args.preferred_source}")
        print(f"Include first frame: {bool(args.include_first_frame)}")
        print(f"Device         : {device}")
        print(f"World size     : {world_size}")
        print(f"Unique pairs   : {overall_pair_stats['shared_total']}")
        if layout == "vbench":
            print(f"Dimension memberships: {overall_pair_stats['dimension_memberships_total']}")
            print(f"Dimensions     : {len(dim_pairs)}")
        print("=================================================")

    overall_result, per_dimension_results, extras = evaluate_unique_dimension_pairs(
        unique_items=unique_dimension_items,
        dim_names=sorted(dim_pairs),
        decoder_backend=decoder_backend,
        device=device,
        frame_batch_size=args.frame_batch_size,
        allow_frame_count_mismatch=args.allow_frame_count_mismatch,
        allow_spatial_mismatch=args.allow_spatial_mismatch,
        include_first_frame=bool(args.include_first_frame),
        metrics=metrics,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        metric_keys=metric_keys,
        decoder=decoder,
        prefetch_depth=args.prefetch_depth,
        temporal=temporal,
        collect_per_video=args.collect_per_video,
    )
    print(f'{overall_result=}')

    failed_pairs = gather_object_lists(extras["failed"], world_size, distributed)
    if rank == 0 and failed_pairs:
        print(
            f"[ERROR] {len(failed_pairs)} video pair(s) were skipped due to errors and excluded "
            "from all averages; see 'failed_pairs' in the report json.",
            flush=True,
        )

    fvd_summary = None
    if temporal is not None and temporal.fvd_enabled:
        feats_baseline = gather_object_lists(extras["fvd_baseline"], world_size, distributed)
        feats_candidate = gather_object_lists(extras["fvd_candidate"], world_size, distributed)
        fvd_num_skipped = all_reduce_int(extras["fvd_skipped"], device, distributed)
        if rank == 0:
            fvd_value = None
            if len(feats_baseline) >= 2:
                fvd_value = temporal_metrics_utils.frechet_distance(
                    np.stack(feats_candidate, axis=0).astype(np.float64),
                    np.stack(feats_baseline, axis=0).astype(np.float64),
                )
            else:
                print(
                    f"[WARN] FVD not computed: only {len(feats_baseline)} videos produced valid "
                    "I3D features (need >= 2).",
                    flush=True,
                )
            fvd_summary = {
                "value": fvd_value,
                "num_videos": len(feats_baseline),
                "num_skipped_videos": fvd_num_skipped,
                "i3d_path": args.i3d_path,
            }

    per_video_records = None
    if args.collect_per_video:
        per_video_records = gather_object_lists(extras["per_video"], world_size, distributed)
        per_video_records.sort(key=lambda rec: rec["key"])
        if rank == 0:
            attach_std_fields(overall_result, per_dimension_results, per_video_records, metric_keys)

    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_root": baseline_root,
        "candidate_root": candidate_root,
        "input_layout": layout,
        "decoder_backend": decoder_backend,
        "decode_device": decoder.device_str if decoder is not None else "cpu",
        "prefetch_depth": args.prefetch_depth,
        "preferred_source": args.preferred_source,
        "device": str(device),
        "world_size": world_size,
        "frame_batch_size": args.frame_batch_size,
        "lpips_net_type": args.lpips_net_type,
        "allow_frame_count_mismatch": args.allow_frame_count_mismatch,
        "allow_spatial_mismatch": args.allow_spatial_mismatch,
        "include_first_frame": bool(args.include_first_frame),
        "collect_per_video": args.collect_per_video,
        "metric_notes": {
            "aggregation": "Each unique video's metric is averaged over aligned frames, then overall results average unique videos referenced by the selected dimensions. Dimension results reuse the same per-video computations.",
            "psnr": "torchmetrics PeakSignalNoiseRatio on RGB frames normalized to [0, 1].",
            "ssim": "torchmetrics StructuralSimilarityIndexMeasure on RGB frames normalized to [0, 1].",
            "lpips": "torchmetrics LearnedPerceptualImagePatchSimilarity with normalize=True, so RGB inputs stay in [0, 1].",
            "include_first_frame": "When false, frame 0 is excluded before frame alignment and metric computation.",
            "psnr_inf_policy": "Pairs containing frames with MSE=0 (PSNR=inf) are marked failed, loudly logged, listed in failed_pairs, and excluded from all averages.",
            "warp_error": "Optional (--compute-warp-error). torchvision RAFT backward flow warps frame t onto t+stride; masked (forward-backward occlusion check) L1 against the real frame. Computed independently for baseline and candidate on the same aligned frames; warp_err_delta = candidate - baseline.",
            "fvd": "Optional (--compute-fvd). Frechet distance between candidate and baseline I3D feature sets (styleganv torchscript recipe); overall only, see the top-level 'fvd' section.",
        },
        "overall_pair_stats": overall_pair_stats,
        "overall_metrics": overall_result,
        "dimension_pair_stats": dim_pair_stats,
        "dimension_metrics": per_dimension_results,
        "failed_pairs": failed_pairs,
    }
    if fvd_summary is not None:
        report["fvd"] = fvd_summary
    if per_video_records is not None:
        report["per_video_metrics"] = per_video_records

    if rank == 0:
        os.makedirs(osp.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report["overall_metrics"], ensure_ascii=False, indent=2))
        print(f"Saved metric summary to: {output_json}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
