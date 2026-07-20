"""FastVAR (ICCV'2025) token pruning ported to the InfinityStar qwen transformer block.

Reproduces the FastVAR baseline used by the FastSTAR paper (layer-wise merging and
unmerging inside the transformer blocks, cf. FastSTAR Appendix B):
- Pivotal Token Selection (PTS): score tokens of the block input by the squared L2
  distance to the global (t*h*w) mean and keep the Top-(L - ratio*L) tokens; with
  `fastvar_per_frame_pts=1` the mean and Top-K are computed within each frame instead.
- 100% pruning (ratio == 1) skips the scale entirely at the model level (see
  FastVAR_InfinityStar.ar_infer_infinity_elegant), matching the original FastVAR
  "skip last scales" behavior.
- Cached Token Restoration (CTR): every prune layer caches its pre-residual
  Attention / FFN outputs at the cache scale (the last full-token scale); at pruned
  scales this cache is interpolated to the current (t, h, w) grid and the computed
  tokens are scattered back, so each block still returns a full-length sequence.

Notes on InfinityStar specifics:
- RoPE for the kept tokens is gathered per batch row before calling the unmodified
  `SelfAttention` (its `apply_rotary_emb` broadcasting is compatible with a
  [2, 1, B, 1, K, half] rope cache).
- KV cache follows the original FastVAR behavior: on a pruned forward that is also
  the `last_repetition_step`, only the kept tokens' K/V persist for later scales.
  With `fastvar_final_iteration_full=1`, multi-repeat scales persist full K/V since
  only the (full-token) final repetition writes the cache.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from models.infinitystar.basic_infinitystar import SelfAttnBlock
from utils.sequence_parallel import SequenceParallelManager as sp_manager


def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x


def pts_select_indices_clip(cur_x, num_remain):
    """Clip-level PTS: squared L2 distance to the mean over all t*h*w tokens, clip-level Top-K."""
    mean_x = cur_x.mean(dim=1, keepdim=True)                                    # [B, 1, C], global avg pool over t*h*w
    mse_difference = torch.sum((cur_x - mean_x) ** 2, dim=-1, keepdim=True)     # [B, L, 1]
    select_indices = torch.argsort(mse_difference, dim=1, descending=True)
    return select_indices[:, :num_remain, :]                                    # [B, num_remain, 1]


def pts_select_indices_per_frame(cur_x, num_remain_per_frame, cur_thw):
    """Per-frame PTS: per-frame mean and per-frame Top-K, so every frame keeps the same token count."""
    B, L, c = cur_x.shape
    t, h, w = cur_thw
    hw = h * w
    frame_x = cur_x.view(B, t, hw, c)
    mean_x = frame_x.mean(dim=2, keepdim=True)                                  # [B, t, 1, C], per-frame avg pool
    mse_difference = torch.sum((frame_x - mean_x) ** 2, dim=-1)                 # [B, t, hw]
    select_indices = torch.argsort(mse_difference, dim=2, descending=True)[:, :, :num_remain_per_frame]
    frame_offsets = torch.arange(t, device=cur_x.device).view(1, t, 1) * hw     # frame-local -> flat token indices
    return (select_indices + frame_offsets).reshape(B, t * num_remain_per_frame, 1)


def masked_previous_scale_cache_3d(cur_x, filted_select_indices, cur_thw, restore_interp_mode='area'):
    """Build PTS merge / CTR unmerge closures for one (t, h, w) scale.

    3D extension of `masked_previous_scale_cache` in models/fastvar/fastvar_basic.py:
    the cached feature map is restored via 3D interpolation.
    """
    B, L, c = cur_x.shape

    def merge(merged_cur_x):
        return torch.gather(merged_cur_x, dim=1, index=filted_select_indices.repeat(1, 1, c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_thw):
        # drop-uncond may shrink the live batch after the cache was written
        unmerged_cache_x_ = unmerged_cache_x[:B]
        unmerged_cache_x_ = unmerged_cache_x_.view(B, *cached_thw, -1).permute(0, 4, 1, 2, 3)   # [B, C, t, h, w]
        unmerged_cache_x_ = F.interpolate(unmerged_cache_x_, size=tuple(cur_thw), mode=restore_interp_mode)
        unmerged_cache_x_ = unmerged_cache_x_.permute(0, 2, 3, 4, 1).reshape(B, L, -1)
        unmerged_cache_x_ = unmerged_cache_x_.to(unmerged_cur_x.dtype)
        unmerged_cache_x_.scatter_(dim=1, index=filted_select_indices.repeat(1, 1, c), src=unmerged_cur_x)
        return unmerged_cache_x_

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx


def compute_merge_3d(
    x: torch.Tensor, prune_ratio: float, x_shape, restore_interp_mode: str = 'area',
    per_frame_pts: bool = False,
) -> Tuple[Callable, Callable, Optional[Callable]]:
    """Return (merge, unmerge, idx_fn); idx_fn is None when pruning degenerates to a no-op."""
    t, h, w = x_shape
    assert x.shape[1] == t * h * w, f'FastVAR token count mismatch: {x.shape[1]} != {t}*{h}*{w}'
    if per_frame_pts:
        r = int(h * w * prune_ratio)
        if r <= 0:
            return do_nothing, do_nothing, None
        filted_select_indices = pts_select_indices_per_frame(x, h * w - r, (t, h, w))
    else:
        r = int(x.shape[1] * prune_ratio)
        if r <= 0:
            return do_nothing, do_nothing, None
        filted_select_indices = pts_select_indices_clip(x, x.shape[1] - r)
    return masked_previous_scale_cache_3d(x, filted_select_indices, (t, h, w), restore_interp_mode)


def gather_rope_cache(rope_cache, keep_indices_Bk1):
    """Gather per-batch-row token RoPE: [2, 1, 1, 1, L, half] -> [2, 1, B, 1, K, half].

    `apply_rotary_emb` drops dim1 and broadcasts the remaining [2, B, 1, K, half]
    cache against q/k shaped [B, H, K, half], so per-row indices stay consistent
    with the per-row PTS token selection.
    """
    B, K = keep_indices_Bk1.shape[0], keep_indices_Bk1.shape[1]
    two, _, _, _, L, half = rope_cache.shape
    rope = rope_cache.expand(two, 1, B, 1, L, half)
    index = keep_indices_Bk1.reshape(1, 1, B, 1, K, 1).expand(two, 1, B, 1, K, half)
    return torch.gather(rope, dim=4, index=index)


class FastVARSelfAttnBlock(SelfAttnBlock):
    """InfinityStar qwen block with FastVAR PTS + CTR around the Attention / FFN ops."""

    def __init__(self, *args, prune_layer: bool = True, restore_interp_mode: str = 'area',
                 per_frame_pts: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx = -1
        self.prune_this_layer = prune_layer
        self.restore_interp_mode = restore_interp_mode
        self.per_frame_pts = per_frame_pts
        # per-step pruning state, injected by FastVAR_InfinityStar.set_fastvar_step_state
        self.fastvar_prune_this_step = False
        self.fastvar_prune_ratio = 0.0
        self.fastvar_x_shape = None             # (t, h, w) of the current scale
        self.fastvar_cache_this_step = False    # True on cache-scale steps
        # CTR caches: pre-residual SA / FFN outputs at the cache scale
        self.previous_scale_cache_self_attn = None
        self.previous_scale_cache_ffn = None
        self.cached_thw = None

    def reset_fastvar_cache(self):
        self.previous_scale_cache_self_attn = None
        self.previous_scale_cache_ffn = None
        self.cached_thw = None

    # NOTE: signature must stay identical to SelfAttnBlock.forward (MultipleLayers calls positionally)
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, rope2d_freqs_grid=[], scale_schedule=[],
                scale_ind=0, context_info=None, last_repetition_step=True, ref_text_scale_inds=[]):
        prune_active = self.fastvar_prune_this_step and self.prune_this_layer
        cache_active = self.fastvar_cache_this_step and self.prune_this_layer
        if not (prune_active or cache_active):
            return super().forward(x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid,
                                   scale_schedule, scale_ind, context_info, last_repetition_step,
                                   ref_text_scale_inds)

        assert not sp_manager.sp_on(), 'FastVAR pruning does not support sequence-parallel inference.'
        if prune_active and (self.previous_scale_cache_self_attn is None or self.previous_scale_cache_ffn is None):
            raise RuntimeError(
                f'FastVAR CTR cache of layer {self.layer_idx} is empty at scale {scale_ind}: '
                f'the cache scale must run before any pruned scale.'
            )

        # ---- self-attention with PTS + CTR ----
        if prune_active:
            merge_fn, unmerge_fn, idx_fn = compute_merge_3d(
                x, self.fastvar_prune_ratio, self.fastvar_x_shape, self.restore_interp_mode,
                per_frame_pts=self.per_frame_pts)
        else:
            merge_fn, unmerge_fn, idx_fn = do_nothing, do_nothing, None
        residual = x
        hidden_states = self.input_layernorm(merge_fn(x))
        rope_cache = rope2d_freqs_grid if idx_fn is None else gather_rope_cache(rope2d_freqs_grid, idx_fn())
        hidden_states = self.attn(hidden_states, attn_bias_or_two_vector, attn_fn, rope_cache, scale_schedule,
                                  scale_ind, context_info, last_repetition_step, ref_text_scale_inds)
        if idx_fn is not None:
            hidden_states = unmerge_fn(hidden_states, self.previous_scale_cache_self_attn, self.cached_thw)
        if cache_active:
            self.previous_scale_cache_self_attn = hidden_states
        hidden_states = residual + hidden_states

        # ---- FFN with PTS + CTR (token selection recomputed on the updated hidden states) ----
        if prune_active:
            merge_fn, unmerge_fn, idx_fn = compute_merge_3d(
                hidden_states, self.fastvar_prune_ratio, self.fastvar_x_shape, self.restore_interp_mode,
                per_frame_pts=self.per_frame_pts)
        else:
            merge_fn, unmerge_fn, idx_fn = do_nothing, do_nothing, None
        residual = hidden_states
        ffn_out = self.mlp(self.post_attention_layernorm(merge_fn(hidden_states)))
        if idx_fn is not None:
            ffn_out = unmerge_fn(ffn_out, self.previous_scale_cache_ffn, self.cached_thw)
        if cache_active:
            self.previous_scale_cache_ffn = ffn_out
            self.cached_thw = tuple(self.fastvar_x_shape)
        hidden_states = residual + ffn_out
        return hidden_states
