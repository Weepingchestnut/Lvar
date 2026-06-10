"""Optional temporal metrics for eval_low_level_metrics.py.

- FVD: I3D features (styleganv `i3d_torchscript.pt`) + Frechet distance,
  following https://github.com/universome/fvd-comparison — the same recipe as
  common_metrics_on_video_quality, so values are comparable with prior work.
  Note: torchmetrics has no FVD implementation, hence this module.
- Flow warping error: per-video temporal consistency. torchvision RAFT flow
  backward-warps frame t onto frame t+stride; the masked L1 between the warp
  and the real frame (occlusions removed via the forward-backward consistency
  check) measures flicker. Reported for baseline and candidate separately,
  plus their delta.

All metric functions consume float RGB videos shaped [T, C, H, W] in [0, 1]
that already live on the metric device — exactly the tensors the pipeline
computes PSNR/SSIM/LPIPS on, after frame alignment and optional resize.
"""

import math
import os.path as osp
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

I3D_DOWNLOAD_HINT = (
    "download `i3d_torchscript.pt` from "
    "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt "
    "(mirror: https://github.com/npurson/fid-metrics/releases/tag/v1.0) "
    "and pass its path via --i3d-path"
)


@dataclass
class TemporalMetricModules:
    i3d: Optional[Any] = None
    raft: Optional[Any] = None
    warp_chunk_size: int = 2
    warp_downscale: float = 1.0
    warp_frame_stride: int = 1

    @property
    def fvd_enabled(self) -> bool:
        return self.i3d is not None

    @property
    def warp_enabled(self) -> bool:
        return self.raft is not None


def build_temporal_modules(
    compute_fvd: bool,
    i3d_path: Optional[str],
    compute_warp_error: bool,
    raft_model: str,
    warp_chunk_size: int,
    warp_downscale: float,
    warp_frame_stride: int,
    device: torch.device,
) -> Optional[TemporalMetricModules]:
    if not compute_fvd and not compute_warp_error:
        return None
    modules = TemporalMetricModules(
        warp_chunk_size=warp_chunk_size,
        warp_downscale=warp_downscale,
        warp_frame_stride=warp_frame_stride,
    )
    if compute_fvd:
        modules.i3d = load_i3d(i3d_path, device)
    if compute_warp_error:
        modules.raft = load_raft(raft_model, device)
    return modules


def load_i3d(i3d_path: Optional[str], device: torch.device):
    if not i3d_path or not osp.isfile(i3d_path):
        raise FileNotFoundError(
            f"I3D torchscript weights not found at: {i3d_path!r}; {I3D_DOWNLOAD_HINT}."
        )
    return torch.jit.load(i3d_path, map_location="cpu").eval().to(device)


def load_raft(raft_model: str, device: torch.device):
    # Weights auto-download to TORCH_HOME on first use; pre-seed the cache on
    # machines without internet access.
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if raft_model == "large":
        model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    else:
        model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    return model.eval().to(device)


@torch.no_grad()
def extract_i3d_features(video: torch.Tensor, i3d) -> Optional[np.ndarray]:
    """I3D feature (400-dim, pre-softmax) of one video; None if too short.

    Mirrors styleganv preprocessing: bilinear resize of the shorter side to
    224, center crop to 224x224, [0,1] -> [-1,1].
    """
    if video.shape[0] < 10:  # I3D downsamples time; <10 frames give no valid feature
        return None
    _, _, h, w = video.shape
    scale = 224.0 / min(h, w)
    if h < w:
        target = (224, math.ceil(w * scale))
    else:
        target = (math.ceil(h * scale), 224)
    frames = F.interpolate(video, size=target, mode="bilinear", align_corners=False)
    h2, w2 = frames.shape[-2:]
    top = (h2 - 224) // 2
    left = (w2 - 224) // 2
    frames = frames[:, :, top : top + 224, left : left + 224]
    clip = (frames * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1, C, T, 224, 224]
    feats = i3d(x=clip, rescale=False, resize=False, return_features=True)
    return feats.squeeze(0).float().cpu().numpy()


def frechet_distance(feats1: np.ndarray, feats2: np.ndarray) -> float:
    """FVD between two feature sets shaped [N, 400] (styleganv formulation)."""
    from scipy.linalg import sqrtm

    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)
    m = np.square(mu1 - mu2).sum()
    if feats1.shape[0] > 1:
        s, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
        fvd = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    else:
        fvd = np.real(m)
    return float(fvd)


def _round_to_multiple_of_8(value: int) -> int:
    # RAFT requires H/W divisible by 8.
    return max(8, int(round(value / 8.0)) * 8)


def _backward_warp(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample `src` at positions displaced by `flow` (flow maps target px -> src px)."""
    _, _, h, w = src.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=src.device, dtype=src.dtype),
        torch.arange(w, device=src.device, dtype=src.dtype),
        indexing="ij",
    )
    coords_x = xx.unsqueeze(0) + flow[:, 0]
    coords_y = yy.unsqueeze(0) + flow[:, 1]
    grid_x = 2.0 * coords_x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * coords_y / max(h - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [B, H, W, 2]
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def compute_warping_error(
    video: torch.Tensor,
    raft,
    chunk_size: int = 2,
    downscale: float = 1.0,
    frame_stride: int = 1,
) -> Optional[float]:
    """Masked L1 warping error of one video (lower = temporally smoother).

    For each frame pair (t, t+stride): RAFT backward flow warps frame t onto
    frame t+stride; pixels failing the forward-backward consistency check
    (Sundaram et al., alpha1=0.01 / alpha2=0.5) count as occluded and are
    excluded from the L1. Returns None when the video has too few frames.
    """
    t = video.shape[0]
    if t < 1 + frame_stride:
        return None
    h, w = video.shape[-2:]
    target_h = _round_to_multiple_of_8(int(h * downscale))
    target_w = _round_to_multiple_of_8(int(w * downscale))
    frames = video
    if (target_h, target_w) != (h, w):
        frames = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)

    src_idx = list(range(0, t - frame_stride, frame_stride))
    dst_idx = [i + frame_stride for i in src_idx]

    err_sum = 0.0
    valid_sum = 0.0
    unmasked_err = 0.0
    pair_count = 0
    for start in range(0, len(src_idx), chunk_size):
        sel_src = src_idx[start : start + chunk_size]
        sel_dst = dst_idx[start : start + chunk_size]
        src = frames[sel_src]
        dst = frames[sel_dst]
        src_in = src * 2.0 - 1.0  # RAFT expects [-1, 1]
        dst_in = dst * 2.0 - 1.0
        flow_bw = raft(dst_in, src_in)[-1]  # flow mapping dst pixels back into src
        flow_fw = raft(src_in, dst_in)[-1]
        warped = _backward_warp(src, flow_bw)
        flow_fw_at_dst = _backward_warp(flow_fw, flow_bw)
        sq_diff = (flow_fw_at_dst + flow_bw).pow(2).sum(dim=1)
        bound = 0.01 * (flow_fw_at_dst.pow(2).sum(dim=1) + flow_bw.pow(2).sum(dim=1)) + 0.5
        mask = (sq_diff < bound).unsqueeze(1).to(src.dtype)  # 1 = non-occluded
        abs_err = (warped - dst).abs()
        err_sum += float((abs_err * mask).sum().item())
        valid_sum += float(mask.sum().item()) * src.shape[1]
        unmasked_err += float(abs_err.mean().item()) * len(sel_src)
        pair_count += len(sel_src)

    if valid_sum > 0:
        return err_sum / valid_sum
    if pair_count > 0:  # fully-occluded edge case: fall back to the unmasked error
        return unmasked_err / pair_count
    return None
