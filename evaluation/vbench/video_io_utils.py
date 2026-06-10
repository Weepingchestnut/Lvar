"""Decode helpers for the low-level metric evaluation pipeline.

Every reader here returns RGB uint8 tensors shaped [T, C, H, W] so the metric
pipeline can skip per-backend color conversion / permutes. TorchCodec decoding
can land directly on GPU (NVDEC); everything else decodes on CPU.

Optional dependencies (torchcodec, torchvision.io) are imported lazily so this
module stays importable in environments that lack them.
"""

import os
import os.path as osp
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Sequence, Tuple

import torch

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class TorchcodecDecoder:
    """torchcodec-based video decoding with sticky CUDA(NVDEC) -> CPU fallback.

    decode_device:
      - "cuda": decode via NVDEC on the metric GPU (requires FFmpeg built with NVDEC).
      - "cpu":  decode on CPU.
      - "auto": try NVDEC when the metric device is CUDA; on the first decode
        failure, log once and permanently fall back to CPU for this process.
    """

    def __init__(self, decode_device: str, metric_device: torch.device, num_ffmpeg_threads: int = 0):
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self._lock = threading.Lock()
        self._warned = False
        if decode_device == "cpu" or metric_device.type != "cuda":
            self._device_str = "cpu"
        else:
            self._device_str = str(metric_device)
        self._allow_fallback = decode_device == "auto"

    @property
    def device_str(self) -> str:
        return self._device_str

    def decode(self, video_path: str) -> torch.Tensor:
        device_str = self._device_str
        try:
            return self._decode_on(video_path, device_str)
        except Exception as exc:
            with self._lock:
                if device_str == "cpu" or not self._allow_fallback:
                    raise
                if not self._warned:
                    print(
                        f"[video_io] GPU(NVDEC) decode failed ({type(exc).__name__}: {exc}); "
                        "falling back to CPU decoding for the rest of this run.",
                        flush=True,
                    )
                    self._warned = True
                self._device_str = "cpu"
            return self._decode_on(video_path, "cpu")

    def _decode_on(self, video_path: str, device_str: str) -> torch.Tensor:
        from torchcodec.decoders import VideoDecoder

        kwargs = {"device": device_str, "dimension_order": "NCHW"}
        if self.num_ffmpeg_threads > 0:
            kwargs["num_ffmpeg_threads"] = self.num_ffmpeg_threads
        try:
            decoder = VideoDecoder(video_path, **kwargs)
        except TypeError:
            # torchcodec API drift: retry with the minimal supported signature.
            decoder = VideoDecoder(video_path, device=device_str)
        frames = decoder[:]  # [T, C, H, W] uint8 RGB on `device_str`
        if frames.numel() == 0:
            raise RuntimeError(f"Decoded zero frames from: {video_path}")
        return frames


def read_frame_dir_rgb(frame_dir: str, max_workers: int = 8, force_cv2: bool = False) -> torch.Tensor:
    """Read a directory of PNG/JPG frames into a [T, C, H, W] RGB uint8 tensor.

    Prefers torchvision.io (decodes straight into CHW torch tensors and releases
    the GIL inside libpng/libjpeg, so a small thread pool overlaps file IO with
    decoding). `force_cv2=True` reproduces the legacy sequential cv2 reader
    (the eval pipeline sets it when the opencv backend is explicitly selected);
    cv2 is also the fallback when torchvision.io is unavailable.

    For 8-bit PNG (lossless, no EXIF) both readers yield bit-identical pixels,
    so switching readers changes speed, not metric values.
    """
    frame_paths = [
        osp.join(frame_dir, name)
        for name in sorted(os.listdir(frame_dir))
        if osp.splitext(name)[1].lower() in _IMAGE_EXTENSIONS
    ]
    if not frame_paths:
        raise RuntimeError(f"Cannot find PNG/JPG frames under: {frame_dir}")

    if not force_cv2:
        try:
            from torchvision.io import ImageReadMode, decode_image
        except ImportError:
            pass
        else:

            def _load(path: str) -> torch.Tensor:
                try:
                    return decode_image(path, mode=ImageReadMode.RGB)  # [C, H, W] uint8
                except TypeError:
                    # torchvision < 0.20: decode_image takes tensors, read_image takes paths.
                    from torchvision.io import read_image

                    return read_image(path, mode=ImageReadMode.RGB)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                frames = list(executor.map(_load, frame_paths))
            return torch.stack(frames, dim=0)

    return _read_frame_dir_rgb_cv2(frame_paths)


def _read_frame_dir_rgb_cv2(frame_paths: Sequence[str]) -> torch.Tensor:
    import cv2
    import numpy as np

    frames = []
    for path in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).contiguous()


def maybe_pin_memory(video: torch.Tensor) -> torch.Tensor:
    """Pin CPU tensors so the later .to(cuda, non_blocking=True) is truly async."""
    if video.device.type != "cpu":
        return video
    try:
        return video.contiguous().pin_memory()
    except RuntimeError:
        return video  # pinning unavailable (e.g. no CUDA); a plain copy still works


class PairPrefetcher:
    """Iterate (local_idx, item, payload), decoding up to `depth` items ahead.

    `payload` is whatever decode_fn(item) returned, or the Exception it raised;
    callers turn decode failures into skipped pairs instead of crashing the run.
    Decoding happens in background threads so it overlaps with the GPU metric
    computation of the current pair.
    """

    def __init__(self, items: Sequence[Any], decode_fn: Callable[[Any], Any], depth: int = 2):
        self._items = list(items)
        self._decode_fn = decode_fn
        self._depth = max(1, int(depth))

    def __iter__(self) -> Iterator[Tuple[int, Any, Any]]:
        executor = ThreadPoolExecutor(max_workers=self._depth)
        pending = deque()
        next_idx = 0
        try:
            while next_idx < len(self._items) and len(pending) < self._depth:
                item = self._items[next_idx]
                pending.append((next_idx, item, executor.submit(self._decode_fn, item)))
                next_idx += 1
            while pending:
                idx, item, future = pending.popleft()
                if next_idx < len(self._items):
                    nxt = self._items[next_idx]
                    pending.append((next_idx, nxt, executor.submit(self._decode_fn, nxt)))
                    next_idx += 1
                try:
                    payload = future.result()
                except Exception as exc:  # surfaced per item; caller decides
                    payload = exc
                yield idx, item, payload
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
