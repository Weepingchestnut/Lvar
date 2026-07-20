"""Pure utility functions for SparseVAR-accelerated InfinityStar inference.

Video (T, H, W) generalization of the image-Infinity SparseVAR mechanism in
models/sparsevar/sparsevar_model.py (lf_anchor / calculate_mse / restore path),
following the FastSTAR baseline protocol: the image reduction mechanism is
applied as-is across the spatiotemporal pyramid, with per-frame spatial anchors
and no temporal awareness. Token order is row-major (t, h, w):
flat index = t * H * W + h * W + w.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def token_mse(x_before: torch.Tensor, x_after: torch.Tensor) -> torch.Tensor:
    """Per-token MSE between two [bs, L, C] feature tensors -> [bs, L]."""
    if x_before.shape != x_after.shape:
        raise ValueError("SparseVAR MSE inputs must have the same shape.")
    return ((x_before - x_after) ** 2).mean(dim=-1)


def average_branch_maps(token_map: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Average a [bs, L] map over CFG branches -> [B, L]; identity when bs == B."""
    bs = token_map.shape[0]
    if bs % batch_size != 0:
        raise ValueError(f"SparseVAR branch map batch {bs} is not a multiple of B={batch_size}.")
    if bs == batch_size:
        return token_map
    return token_map.reshape(bs // batch_size, batch_size, -1).mean(0)


def dynamic_keep_ratio(mse_BL: torch.Tensor, threshold: float) -> float:
    """Dynamic keep ratio from the MSE map (port of sparsevar_model.py:466)."""
    return float(((mse_BL / mse_BL.max()) > threshold).float().mean(1).max())


def upsample_token_map(flat: torch.Tensor, prev_pn, cur_pn) -> torch.Tensor:
    """Nearest-upsample a [B, L'] or [B, L', C] token map from prev_pn to cur_pn."""
    squeeze_channel = flat.dim() == 2
    if squeeze_channel:
        flat = flat.unsqueeze(-1)
    B, L, C = flat.shape
    t0, h0, w0 = (int(v) for v in prev_pn)
    t1, h1, w1 = (int(v) for v in cur_pn)
    if L != t0 * h0 * w0:
        raise ValueError(f"SparseVAR token map length {L} does not match prev_pn {tuple(prev_pn)}.")
    volume = flat.reshape(B, t0, h0, w0, C).permute(0, 4, 1, 2, 3)
    volume = F.interpolate(volume, size=(t1, h1, w1), mode='nearest')
    out = volume.permute(0, 2, 3, 4, 1).reshape(B, t1 * h1 * w1, C)
    return out.squeeze(-1) if squeeze_channel else out


def build_frame_anchor_coords(H: int, W: int, window: int) -> torch.Tensor:
    """Per-frame spatial anchor (h, w) coords on a window x window grid -> [A, 2].

    Rectangular generalization of generate_anchors_from_flat
    (sparsevar_model.py:92-106), keeping the append-last-row/col semantics.
    """
    hs = torch.arange(0, H, window)
    if (H - 1) % window != 0:
        hs = torch.cat([hs, torch.tensor([H - 1], dtype=hs.dtype)])
    ws = torch.arange(0, W, window)
    if (W - 1) % window != 0:
        ws = torch.cat([ws, torch.tensor([W - 1], dtype=ws.dtype)])
    grid_h, grid_w = torch.meshgrid(hs, ws, indexing='ij')
    return torch.stack([grid_h.flatten(), grid_w.flatten()], dim=-1)


def nearest_anchor_ids(H: int, W: int, anchor_coords: torch.Tensor, num_candidates: int = 4) -> torch.Tensor:
    """Indices (into the anchor list) of the nearest anchors per (h, w) position -> [H*W, k].

    Purely geometric and frame-independent (port of map_to_best_corner's
    distance sort, sparsevar_model.py:47-54), so it is computed once per scale.
    """
    pos = torch.arange(H * W, device=anchor_coords.device)
    i, j = pos // W, pos % W
    dis = (i[:, None] - anchor_coords[None, :, 0]) ** 2 + (j[:, None] - anchor_coords[None, :, 1]) ** 2
    k = min(int(num_candidates), anchor_coords.shape[0])
    return dis.sort(dim=1)[1][:, :k]


def match_tokens_to_anchors(
    feat_BLV: torch.Tensor,
    cur_pn,
    anchor_coords: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best anchor per token via cosine similarity (port of sparsevar_model.py:56-75).

    Returns (best_anchor_flat [B, L] with t*H*W frame offsets, max_sim [B, L]).
    """
    B, L, V = feat_BLV.shape
    t, h, w = (int(v) for v in cur_pn)
    HW = h * w
    if L != t * HW:
        raise ValueError(f"SparseVAR feature length {L} does not match cur_pn {tuple(cur_pn)}.")

    feat = feat_BLV.reshape(B, t, HW, V)
    anchor_flat_hw = anchor_coords[:, 0] * w + anchor_coords[:, 1]                # [A]
    anchor_feats = feat[:, :, anchor_flat_hw]                                     # [B, T, A, V]
    cand_feats = anchor_feats[:, :, candidate_ids]                                # [B, T, HW, k, V]
    similarities = F.cosine_similarity(feat.unsqueeze(3), cand_feats, dim=-1)     # [B, T, HW, k]

    best = similarities.argmax(dim=-1)                                            # [B, T, HW]
    cand_pos_hw = anchor_flat_hw[candidate_ids]                                   # [HW, k]
    best_pos_hw = torch.gather(
        cand_pos_hw.view(1, 1, HW, -1).expand(B, t, -1, -1), 3, best.unsqueeze(-1)
    ).squeeze(-1)                                                                 # [B, T, HW]

    frame_offsets = (torch.arange(t, device=feat.device) * HW).view(1, t, 1)
    best_anchor_flat = (best_pos_hw + frame_offsets).reshape(B, L)
    max_sim = similarities.max(dim=-1)[0].reshape(B, L)
    return best_anchor_flat, max_sim


def get_reverse_indices(index: torch.Tensor, L: int) -> torch.Tensor:
    """Complement of the kept-index set (port of sparsevar_model.py:78-89)."""
    B = index.size(0)
    all_indices = torch.arange(L, device=index.device).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, L, dtype=torch.bool, device=index.device)
    mask.scatter_(1, index, False)
    return all_indices[mask].view(B, -1)


def lf_anchor_video(
    prev_logits_BLV: torch.Tensor,
    mse_prev_BL: torch.Tensor,
    prev_pn,
    cur_pn,
    keep_ratio: float,
    window: int,
    beta: float,
    force_keep_anchors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Video port of SparseVAR's lf_anchor (sparsevar_model.py:236-293).

    force_keep_anchors=True follows the paper's Algorithm 1 line 8 ("Exclude
    tokens in M_low except anchor tokens"): the per-frame anchor lattice is
    always computed. Anchors are carved out of the keep budget (not added on
    top), so the kept-token count still equals keep_ratio * L and latency
    stays comparable across both settings. False reproduces the released
    image code's effective behavior, whose anchor-merge branch never triggers
    on the post-CFG B=1 path.

    Returns (high_freq_idx [B, K], low_freq_idx [B, L-K],
             recoverable_mask [B, L-K] bool, low_to_anchor_idx [B, L-K]).
    """
    t, h, w = (int(v) for v in cur_pn)
    L = t * h * w

    upsampled_logits = upsample_token_map(prev_logits_BLV, prev_pn, cur_pn)   # [B, L, V]
    upsampled_mse = upsample_token_map(mse_prev_BL, prev_pn, cur_pn)          # [B, L]
    device = upsampled_logits.device

    # Per-frame spatial anchor lattice.
    anchor_coords = build_frame_anchor_coords(h, w, window).to(device)

    # High/low-frequency split: larger MSE change => high-frequency (kept).
    num_keep = max(1, min(L, int(L * float(keep_ratio))))
    if force_keep_anchors:
        anchor_flat_hw = anchor_coords[:, 0] * w + anchor_coords[:, 1]                  # [A]
        frame_offsets = torch.arange(t, device=device) * (h * w)
        anchor_global = (frame_offsets[:, None] + anchor_flat_hw[None, :]).reshape(-1)  # [t*A]
        B = upsampled_mse.shape[0]
        num_metric = num_keep - anchor_global.numel()
        if num_metric > 0:
            anchor_mask = torch.zeros(L, dtype=torch.bool, device=device)
            anchor_mask[anchor_global] = True
            pool = upsampled_mse.masked_fill(anchor_mask[None, :], float('-inf'))
            metric_idx = pool.topk(num_metric, dim=1).indices                           # [B, num_metric]
            high_freq_idx = torch.cat(
                [anchor_global[None, :].expand(B, -1), metric_idx], dim=1)
        else:
            # Anchor lattice alone exceeds the keep budget; keep all anchors
            # (realized keep ratio then exceeds the nominal one).
            high_freq_idx = anchor_global[None, :].expand(B, -1)
        high_freq_idx = high_freq_idx.sort(dim=1).values
    else:
        high_freq_idx = upsampled_mse.sort(1)[1][:, -num_keep:]
    low_freq_idx = get_reverse_indices(high_freq_idx, L)

    candidate_ids = nearest_anchor_ids(h, w, anchor_coords)
    best_anchor_flat, max_sim = match_tokens_to_anchors(
        upsampled_logits, cur_pn, anchor_coords, candidate_ids)

    recoverable_mask = max_sim.gather(dim=1, index=low_freq_idx) > beta
    low_to_anchor_idx = best_anchor_flat.gather(dim=1, index=low_freq_idx)
    return high_freq_idx, low_freq_idx, recoverable_mask, low_to_anchor_idx


def restore_pruned_tokens(
    x_origin: torch.Tensor,
    computed: torch.Tensor,
    high_freq_idx: torch.Tensor,
    low_freq_idx: torch.Tensor,
    recoverable_mask: torch.Tensor,
    low_to_anchor_idx: torch.Tensor,
) -> torch.Tensor:
    """Scatter computed tokens back and copy anchor outputs to recoverable
    low-frequency tokens (port of sparsevar_model.py:468-477).

    Non-recoverable low-frequency positions keep their pre-block input
    embedding; their sampled codes are zeroed later by zero_nonrecoverable_codes.
    """
    bs, L, C = x_origin.shape
    if bs % high_freq_idx.shape[0] != 0:
        raise ValueError(f"SparseVAR restore batch {bs} is not a multiple of {high_freq_idx.shape[0]}.")
    rep = bs // high_freq_idx.shape[0]

    x_origin = x_origin.to(computed)
    hr_expand = high_freq_idx.repeat(rep, 1).unsqueeze(-1).expand(-1, -1, C)
    x_full = x_origin.scatter(1, hr_expand, computed)

    lr_expand = low_freq_idx.repeat(rep, 1).unsqueeze(-1).expand(-1, -1, C)
    anchor_expand = low_to_anchor_idx.repeat(rep, 1).unsqueeze(-1).expand(-1, -1, C)
    source_flag = torch.gather(x_full, dim=1, index=anchor_expand)
    dst = torch.gather(x_full, dim=1, index=lr_expand)
    recover = recoverable_mask.to(dst.dtype).repeat(rep, 1).unsqueeze(-1)
    source_flag = dst * (1 - recover) + recover * source_flag
    return x_full.scatter(1, lr_expand, source_flag)


def zero_nonrecoverable_codes(
    codes: torch.Tensor,
    low_freq_idx: torch.Tensor,
    recoverable_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero latent codes at non-recoverable low-frequency positions
    (port of sparsevar_model.py:511-530, non-patchify branch).

    codes: [B, C, T, H, W]; the (T, H, W) grid must match the token grid the
    indices were built on (InfinityStar 480p runs apply_spatial_patchify=0).
    """
    if codes.dim() != 5:
        raise ValueError("SparseVAR expects [B, C, T, H, W] latent codes.")
    B = codes.shape[0]
    L = codes.shape[2] * codes.shape[3] * codes.shape[4]
    if low_freq_idx.shape[0] != B:
        raise ValueError(
            f"SparseVAR codes batch {B} does not match token plan batch {low_freq_idx.shape[0]}.")

    zero_mask = torch.zeros(B, L, dtype=torch.bool, device=codes.device)
    zero_mask.scatter_(1, low_freq_idx, ~recoverable_mask)
    codes_flat = codes.reshape(B, codes.shape[1], L)
    codes_flat = codes_flat.masked_fill(zero_mask[:, None, :], 0)
    return codes_flat.view_as(codes)
