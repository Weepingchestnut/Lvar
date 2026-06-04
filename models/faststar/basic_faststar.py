import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _resize_feature(feature: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(feature.shape[-3:]) == tuple(size):
        return feature
    return F.interpolate(feature, size=size, mode="trilinear", align_corners=False)


def spatial_cosine_similarity(
    previous_feature: torch.Tensor,
    current_feature: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute cross-scale cosine similarity for [B, C, T, H, W] features."""
    if previous_feature.dim() != 5 or current_feature.dim() != 5:
        raise ValueError("FastSTAR expects [B, C, T, H, W] feature tensors.")
    previous_feature = _resize_feature(previous_feature, current_feature.shape[-3:])
    return F.cosine_similarity(previous_feature.float(), current_feature.float(), dim=1, eps=eps).clamp(-1, 1)


def temporal_cosine_similarity(
    current_feature: torch.Tensor,
    previous_temporal_feature: Optional[torch.Tensor] = None,
    fallback: str = "spatial_only",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute adjacent-frame cosine similarity for [B, C, T, H, W] features."""
    if current_feature.dim() != 5:
        raise ValueError("FastSTAR expects [B, C, T, H, W] feature tensors.")
    bsz, _, frames, height, width = current_feature.shape
    sim = torch.ones((bsz, frames, height, width), device=current_feature.device, dtype=torch.float32)

    if previous_temporal_feature is not None:
        prev = _resize_feature(previous_temporal_feature[:, :, -1:], (1, height, width))
        sim[:, 0] = F.cosine_similarity(prev.float(), current_feature[:, :, :1].float(), dim=1, eps=eps).squeeze(1).clamp(-1, 1)
    elif fallback not in {"spatial_only", "all_ones"}:
        raise ValueError(f"Unsupported FastSTAR temporal fallback: {fallback}")

    if frames > 1:
        sim[:, 1:] = F.cosine_similarity(
            current_feature[:, :, :-1].float(),
            current_feature[:, :, 1:].float(),
            dim=1,
            eps=eps,
        ).clamp(-1, 1)
    return sim


def compute_st_score(
    previous_feature: torch.Tensor,
    current_feature: torch.Tensor,
    previous_temporal_feature: Optional[torch.Tensor] = None,
    p_norm: float = 2.0,
    temporal_fallback: str = "spatial_only",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fuse spatial and temporal dissimilarity into the FastSTAR ST score."""
    spatial_similarity = spatial_cosine_similarity(previous_feature, current_feature, eps=eps)
    temporal_similarity = temporal_cosine_similarity(
        current_feature,
        previous_temporal_feature=previous_temporal_feature,
        fallback=temporal_fallback,
        eps=eps,
    )
    spatial_dissimilarity = 1.0 - spatial_similarity
    temporal_dissimilarity = 1.0 - temporal_similarity
    if math.isinf(float(p_norm)):
        return torch.maximum(spatial_dissimilarity, temporal_dissimilarity)
    return (spatial_dissimilarity.pow(p_norm) + temporal_dissimilarity.pow(p_norm)).pow(1.0 / p_norm)


def topk_pruning_mask(
    st_score: torch.Tensor,
    prune_ratio: float,
    per_frame_topk: bool = True,
) -> torch.Tensor:
    """Return a bool keep mask shaped [B, T, H, W] using top-k ST scores."""
    if st_score.dim() != 4:
        raise ValueError("FastSTAR ST score must be shaped [B, T, H, W].")
    prune_ratio = min(max(float(prune_ratio), 0.0), 1.0)
    keep_ratio = 1.0 - prune_ratio
    if keep_ratio >= 1.0:
        return torch.ones_like(st_score, dtype=torch.bool)

    bsz, frames, height, width = st_score.shape
    if per_frame_topk:
        flat = st_score.reshape(bsz * frames, height * width)
        keep_k = max(1, math.ceil(flat.shape[-1] * keep_ratio))
        keep_idx = flat.topk(keep_k, dim=-1).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(1, keep_idx, True)
        return mask.reshape(bsz, frames, height, width)

    flat = st_score.reshape(bsz, frames * height * width)
    keep_k = max(1, math.ceil(flat.shape[-1] * keep_ratio))
    keep_idx = flat.topk(keep_k, dim=-1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, keep_idx, True)
    return mask.reshape(bsz, frames, height, width)


def resize_pruning_mask(pruning_mask: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    """Resize a [B, T, H, W] bool mask with nearest interpolation."""
    if pruning_mask.dim() != 4:
        raise ValueError("FastSTAR pruning mask must be shaped [B, T, H, W].")
    if tuple(pruning_mask.shape[-3:]) == tuple(size):
        return pruning_mask.bool()
    mask = pruning_mask[:, None].float()
    mask = F.interpolate(mask, size=size, mode="nearest")
    return mask[:, 0] >= 0.5


def partial_update(
    previous_feature: torch.Tensor,
    residual: torch.Tensor,
    pruning_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply FastSTAR partial update, leaving pruned regions equal to previous_feature."""
    if previous_feature.shape != residual.shape:
        residual = _resize_feature(residual, previous_feature.shape[-3:])
    mask = resize_pruning_mask(pruning_mask, previous_feature.shape[-3:])[:, None].to(residual.dtype)
    return previous_feature + residual * mask


def _pruning_mask_contact_sheet(pruning_mask: torch.Tensor, frame_gap: int = 2) -> torch.Tensor:
    """Convert a [B, T, H, W] keep mask into a grayscale temporal contact sheet."""
    if pruning_mask.dim() != 4:
        raise ValueError("FastSTAR pruning mask must be shaped [B, T, H, W].")
    mask = pruning_mask.detach().bool().cpu().to(torch.uint8) * 255
    bsz, frames, height, width = mask.shape
    if bsz <= 0 or frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("FastSTAR pruning mask must have non-empty B/T/H/W dimensions.")

    cols = max(1, math.ceil(math.sqrt(frames)))
    rows = math.ceil(frames / cols)
    sheet_height = rows * height + max(0, rows - 1) * frame_gap
    sheet_width = cols * width + max(0, cols - 1) * frame_gap

    batch_sheets = []
    for batch_idx in range(bsz):
        sheet = torch.zeros((sheet_height, sheet_width), dtype=torch.uint8)
        for frame_idx in range(frames):
            row_idx, col_idx = divmod(frame_idx, cols)
            top = row_idx * (height + frame_gap)
            left = col_idx * (width + frame_gap)
            sheet[top:top + height, left:left + width] = mask[batch_idx, frame_idx]
        batch_sheets.append(sheet)

    if len(batch_sheets) == 1:
        return batch_sheets[0]

    batch_gap = torch.zeros((frame_gap, sheet_width), dtype=torch.uint8)
    stacked = []
    for batch_idx, sheet in enumerate(batch_sheets):
        if batch_idx > 0 and frame_gap > 0:
            stacked.append(batch_gap)
        stacked.append(sheet)
    return torch.cat(stacked, dim=0)


def save_pruning_mask_visualization(
    pruning_mask: torch.Tensor,
    save_dir: str,
    scale_index: int,
    repeat_index: int,
    upscale: int = 4,
) -> str:
    """Save a PNG visualization of a FastSTAR pruning keep mask."""
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"scale_{scale_index:02d}_repeat{repeat_index}.png")
    sheet = _pruning_mask_contact_sheet(pruning_mask)
    image = Image.fromarray(sheet.numpy())
    upscale = max(1, int(upscale))
    if upscale > 1:
        resample_nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        image = image.resize((image.width * upscale, image.height * upscale), resample=resample_nearest)
    image.save(save_path)
    return save_path


def save_pruning_mask(pruning_mask: torch.Tensor, save_dir: str, scale_index: int, repeat_index: int) -> str:
    """Save a raw FastSTAR keep mask and a matching PNG visualization."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"scale_{scale_index:02d}_repeat{repeat_index}.pt")
    torch.save(pruning_mask.detach().cpu(), save_path)
    save_pruning_mask_visualization(pruning_mask, save_dir, scale_index, repeat_index)
    return save_path
