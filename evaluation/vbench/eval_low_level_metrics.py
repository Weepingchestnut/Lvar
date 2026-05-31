import argparse
import cv2
import json
import os
import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from utils.misc import time_str


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
        description="Evaluate low-level video metrics between baseline and accelerated VBench generations."
    )
    parser.add_argument("--baseline-root", type=str, required=True, help="Baseline VBench result root.")
    parser.add_argument("--candidate-root", type=str, required=True, help="Accelerated-model VBench result root.")
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
        choices=("auto", "decord", "opencv"),
        default="auto",
        help="Video decoding backend. 'auto' prefers decord and falls back to opencv.",
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
        raise ImportError("Neither decord nor opencv-python is available for video decoding.") from exc


def read_frame_dir_rgb(frame_dir: str) -> torch.Tensor:
    # import cv2

    frame_names = [
        name for name in sorted(os.listdir(frame_dir))
        if osp.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not frame_names:
        raise RuntimeError(f"Cannot find PNG/JPG frames under: {frame_dir}")

    frames = []
    for frame_name in frame_names:
        frame_path = osp.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return torch.from_numpy(np.stack(frames, axis=0))


def read_video_rgb(video_path: str, backend: str) -> torch.Tensor:
    if osp.isdir(video_path):
        return read_frame_dir_rgb(video_path)

    suffix = osp.splitext(video_path)[1].lower()
    if suffix == ".npy":
        frames = np.load(video_path)
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim != 4:
            raise RuntimeError(f"Expected npy frames with shape [T,H,W,3], got {frames.shape}: {video_path}")
        return torch.from_numpy(frames)

    if suffix in IMAGE_EXTENSIONS:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required to read image files.") from exc
        image = cv2.imread(video_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {video_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image[None, ...])

    if backend == "decord":
        # import cv2
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        frame_ids = list(range(len(vr)))
        frames = vr.get_batch(frame_ids).asnumpy()
        return torch.from_numpy(frames)

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
    return torch.from_numpy(np.stack(frames, axis=0))


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
    return video.permute(0, 3, 1, 2).contiguous().float().div_(255.0).to(device, non_blocking=True)


def maybe_resize_candidate(candidate: torch.Tensor, baseline_hw: Tuple[int, int]) -> torch.Tensor:
    if tuple(candidate.shape[-2:]) == baseline_hw:
        return candidate
    return torch.nn.functional.interpolate(
        candidate,
        size=baseline_hw,
        mode="bilinear",
        align_corners=False,
    )


def evaluate_video_pair(
    pair: VideoPair,
    decoder_backend: str,
    device: torch.device,
    frame_batch_size: int,
    allow_frame_count_mismatch: bool,
    allow_spatial_mismatch: bool,
    include_first_frame: bool,
    metrics,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Given a pair of baseline/candidate videos, return the average PSNR/SSIM/LPIPS of this pair of videos.

    Args:
        pair (VideoPair): _description_
        decoder_backend (str): _description_
        device (torch.device): _description_
        frame_batch_size (int): _description_
        allow_frame_count_mismatch (bool): _description_
        allow_spatial_mismatch (bool): _description_
        metrics (_type_): _description_

    Returns:
        Tuple[Optional[Dict[str, float]], Optional[str]]: _description_
    """
    psnr_metric, ssim_metric, lpips_metric = metrics

    baseline_frames = read_video_rgb(pair.baseline_path, decoder_backend)       # torch.Size([81, 720, 1280, 3])
    candidate_frames = read_video_rgb(pair.candidate_path, decoder_backend)

    if baseline_frames.ndim != 4 or candidate_frames.ndim != 4:
        return None, f"decoded tensor rank mismatch for {pair.key}"

    if not include_first_frame:
        if baseline_frames.shape[0] <= 1 or candidate_frames.shape[0] <= 1:
            return None, f"cannot exclude first frame for {pair.key}: not enough frames"
        baseline_frames = baseline_frames[1:]
        candidate_frames = candidate_frames[1:]

    baseline_t, baseline_h, baseline_w = baseline_frames.shape[:3]
    candidate_t, candidate_h, candidate_w = candidate_frames.shape[:3]

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

    baseline_frames = move_video_to_device(baseline_frames, device)         # torch.Size([81, 3, 720, 1280])
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
            batch_size = baseline_chunk.shape[0]

            psnr_sum += float(psnr_metric(candidate_chunk, baseline_chunk).item())
            psnr_metric.reset()

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


def make_meter_set() -> Dict[str, RunningMean]:
    return {
        "psnr": RunningMean(),
        "ssim": RunningMean(),
        "lpips": RunningMean(),
    }


def update_meter_set(meters: Dict[str, RunningMean], result: Dict[str, float]) -> None:
    meters["psnr"].update(result["psnr"], 1)
    meters["ssim"].update(result["ssim"], 1)
    meters["lpips"].update(result["lpips"], 1)


def reduce_meter_set(
    meters: Dict[str, RunningMean],
    local_errors: int,
    device: torch.device,
    distributed: bool,
) -> Dict[str, Optional[float]]:
    global_psnr_sum = all_reduce_float(meters["psnr"].value_sum, device, distributed)
    global_ssim_sum = all_reduce_float(meters["ssim"].value_sum, device, distributed)
    global_lpips_sum = all_reduce_float(meters["lpips"].value_sum, device, distributed)
    global_count = all_reduce_int(meters["psnr"].count, device, distributed)
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
) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Optional[float]]]]:
    local_items = list(unique_items[rank::world_size])
    overall_meters = make_meter_set()
    dim_meters = {dim: make_meter_set() for dim in dim_names}
    local_overall_errors = 0
    local_dim_errors = {dim: 0 for dim in dim_names}

    for local_idx, item in enumerate(local_items):
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
            print(f"[Rank {rank}] Skip unique pair {item.pair.key}: {error}")
            continue

        assert result is not None
        update_meter_set(overall_meters, result)
        for dim in item.dimensions:
            update_meter_set(dim_meters[dim], result)

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
    return overall_result, per_dimension_results


def default_output_json(candidate_root: str) -> str:
    save_dir = osp.join(candidate_root, "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)
    return osp.join(save_dir, f"low_level_metrics_vs_baseline_{time_str()}.json")


def main() -> None:
    args = parse_args()
    device, rank, world_size, distributed = maybe_init_distributed(args.device)

    baseline_root, baseline_videos_dir, baseline_dims_dir, baseline_frames_dims_dir = resolve_vbench_dirs(args.baseline_root)
    candidate_root, candidate_videos_dir, candidate_dims_dir, candidate_frames_dims_dir = resolve_vbench_dirs(args.candidate_root)
    output_json = osp.abspath(args.output_json) if args.output_json else default_output_json(candidate_root)

    decoder_backend = choose_decoder_backend(args.decode_backend)
    metrics = build_metric_modules(device=device, lpips_net_type=args.lpips_net_type)

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

    if rank == 0:
        print("========== Low-Level Metric Evaluation ==========")
        print(f"Baseline root  : {baseline_root}")
        print(f"Candidate root : {candidate_root}")
        print(f"Decoder backend: {decoder_backend}")
        print(f"Preferred source: {args.preferred_source}")
        print(f"Include first frame: {bool(args.include_first_frame)}")
        print(f"Device         : {device}")
        print(f"World size     : {world_size}")
        print(f"Unique pairs   : {overall_pair_stats['shared_total']}")
        print(f"Dimension memberships: {overall_pair_stats['dimension_memberships_total']}")
        print(f"Dimensions     : {len(dim_pairs)}")
        print("=================================================")

    overall_result, per_dimension_results = evaluate_unique_dimension_pairs(
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
    )
    print(f'{overall_result=}')

    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_root": baseline_root,
        "candidate_root": candidate_root,
        "decoder_backend": decoder_backend,
        "preferred_source": args.preferred_source,
        "device": str(device),
        "world_size": world_size,
        "frame_batch_size": args.frame_batch_size,
        "lpips_net_type": args.lpips_net_type,
        "allow_frame_count_mismatch": args.allow_frame_count_mismatch,
        "allow_spatial_mismatch": args.allow_spatial_mismatch,
        "include_first_frame": bool(args.include_first_frame),
        "metric_notes": {
            "aggregation": "Each unique video's metric is averaged over aligned frames, then overall results average unique videos referenced by the selected dimensions. Dimension results reuse the same per-video computations.",
            "psnr": "torchmetrics PeakSignalNoiseRatio on RGB frames normalized to [0, 1].",
            "ssim": "torchmetrics StructuralSimilarityIndexMeasure on RGB frames normalized to [0, 1].",
            "lpips": "torchmetrics LearnedPerceptualImagePatchSimilarity with normalize=True, so RGB inputs stay in [0, 1].",
            "include_first_frame": "When false, frame 0 is excluded before frame alignment and metric computation.",
        },
        "overall_pair_stats": overall_pair_stats,
        "overall_metrics": overall_result,
        "dimension_pair_stats": dim_pair_stats,
        "dimension_metrics": per_dimension_results,
    }

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
