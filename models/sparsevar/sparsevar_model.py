import math
import random
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

import utils.dist as dist
from models.infinity.basic_infinity import (AdaLNBeforeHead, FastRMSNorm,
                                            SelfAttnBlock, flash_attn_func,
                                            flash_fused_op_installed,
                                            precompute_rope2d_freqs_grid)
from models.infinity.flex_attn import FlexAttn
from models.infinity.infinity_model import (
    TIMM_KEYS, Infinity, MultiInpIdentity, MultipleLayers, SharedAdaLin,
    sample_with_top_k_top_p_also_inplace_modifying_logits_)
from models.sparsevar.basic_sparsevar import CrossAttention, CrossAttnBlock
from utils.dist import for_visualize
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from models.infinity.fused_op import (fused_ada_layer_norm,
                                          fused_ada_rms_norm)
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None


def do_nothing(x, mode=None):
    return x


def map_to_best_corner(N, n, pos, matrix, anchor_indices):
    # Convert flattened positions to row/col indices
    i = pos // N
    j = pos % N
    batch_idx = torch.arange(matrix.shape[0]).unsqueeze(1).expand(-1, pos.shape[1]).to(matrix.device)

    # Gather feature vectors at query positions
    features_at_pos = matrix[batch_idx, i, j, :]  # (B, T, C)

    # Compute squared distance from each pos to candidate anchors
    dis = (i[:, None, :] - anchor_indices[..., None, 0].to(matrix.device)) ** 2 + \
          (j[:, None, :] - anchor_indices[..., None, 1].to(matrix.device)) ** 2

    # Pick 4 nearest anchors for each position
    closest_pos = dis.sort(1)[1][:, :4]
    anchor_positions = anchor_indices.unsqueeze(1).expand(-1, N ** 2, -1, -1) \
        .gather(index=closest_pos.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 2), dim=1)

    # Gather anchor features
    B_idx = torch.arange(anchor_positions.shape[0]).unsqueeze(1).unsqueeze(1).expand(
        -1, anchor_positions.shape[1], 4
    ).to(anchor_positions.device)
    corner_features = matrix[B_idx, anchor_positions[..., 0], anchor_positions[..., 1], :]  # (B, T, 4, C)

    # Cosine similarity between pos features and corner features
    similarities = F.cosine_similarity(features_at_pos.unsqueeze(2), corner_features, dim=-1)  # (B, T, 4)

    # Pick best anchor per position
    best_indices = torch.argmax(similarities, dim=-1)  # (B, T)
    best_pos = anchor_positions.gather(
        dim=2,
        index=best_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2)
    )[:, :, 0]

    # Convert (row, col) back to flat index
    best_pos = best_pos[..., 0] * N + best_pos[..., 1]

    return best_pos, similarities


def get_reverse_indices(index, L):
    B = index.size(0)
    all_indices = torch.arange(L).unsqueeze(0).expand(B, -1).to(index.device)

    # Mask out the given indices
    mask = torch.ones(B, L, dtype=torch.bool, device=index.device)
    mask.scatter_(1, index, False)

    # Select remaining indices
    remaining_indices = all_indices[mask].view(B, -1)

    return remaining_indices


def generate_anchors_from_flat(N, n):
    # Generate coordinates every n steps
    indices = torch.arange(0, N, n)

    # Ensure the last position (N-1) is included if not divisible
    if n % N != 0:
        indices = torch.cat([indices, torch.tensor([N - 1]).to(indices)])

    # Create row/col meshgrid
    grid_x, grid_y = torch.meshgrid(indices, indices, indexing='ij')

    # Flatten into (num_anchors, 2)
    anchors = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    return anchors


def compute_avg_similarity(feature_map, kernel_size):
    B, H, W, C = feature_map.shape

    # Extract local patches using unfold
    unfold = F.unfold(
        feature_map.permute(0, 3, 1, 2),  # (B, C, H, W)
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2
    )
    # Reshape to (B, C, K*K, H*W)
    unfolded_neighborhood = unfold.view(B, C, kernel_size ** 2, H * W)

    # Flatten original features to (B, C, H*W)
    center_features = feature_map.permute(0, 3, 1, 2).flatten(2)

    # Compute cosine similarity with neighbors, then average
    avg_similarity_matrix = torch.nn.functional.cosine_similarity(
        center_features.unsqueeze(-2).expand(-1, -1, kernel_size ** 2, -1),
        unfolded_neighborhood,
        dim=1
    ).mean(1).reshape(B, H, W)

    return avg_similarity_matrix


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)

    def forward(self, ca_kv):
        output, _ = self.ca(None, ca_kv)
        return output.squeeze(1)


def calculate_mse(input_tensor, output_tensor):
    if input_tensor.shape != output_tensor.shape:
        raise ValueError("Input and output tensors must have the same shape.")
    
    mse_per_position = ((input_tensor - output_tensor) ** 2).mean(dim=-1)
    return mse_per_position


def select_and_set_zero(tokens,hr_idx):
    tokens_=torch.zeros_like(tokens)
    hr_idx_expanded = hr_idx.unsqueeze(-1).repeat(tokens.shape[0]//hr_idx.shape[0], 1, tokens.shape[-1])
    gathered_values = torch.gather(tokens, 1, index=hr_idx_expanded)
    tokens_.scatter_(1, hr_idx_expanded, gathered_values)
    return tokens_


@register_model
def sparsevar_infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8, **kwargs):
    return SparseVAR_Infinity(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        block_chunks=block_chunks,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def sparsevar_infinity_8b(depth=40, embed_dim=3584, num_heads=3584//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8, **kwargs):
    return SparseVAR_Infinity(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        block_chunks=block_chunks,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )


class SparseVAR_Infinity(Infinity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.compress_method = getattr(self.other_args, 'sparsevar_compress_method', None)
        self.compress_ratio = getattr(self.other_args, 'sparsevar_compress_ratio', 0.6)
        self.local_window_size = getattr(self.other_args, 'sparsevar_local_window_size', 4)
        self.start_prune_stage = getattr(self.other_args, 'sparsevar_start_prune_stage', 10)
        # use No.3 chunk to get MSE
        self.specific_mse_layer = getattr(self.other_args, 'sparsevar_specific_mse_layer', 3)
        self.beta = getattr(self.other_args, 'sparsevar_beta', 0.8)

        self.unregistered_blocks = []
        for block_idx in range(self.depth):
            block = (CrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=self.cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=self.shared_aln, norm_layer=self.norm_layer,
                num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, drop=self.drop_rate, drop_path=self.dpr[block_idx], tau=self.tau, cos_attn=self.cos_attn,
                swiglu=self.swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=self.fused_mlp, fused_norm_func=self.fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=self.use_flex_attn, batch_size=self.batch_size, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)

        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(
                    self.unregistered_blocks,
                    self.num_blocks_in_a_chunk,
                    i*self.num_blocks_in_a_chunk
                ))
        
        print(f"\nSparseVAR (ICCV'2025)")
        print(
            f'    Compress Method: {self.compress_method}\n',
            f'    Compress Ratio: {self.compress_ratio}\n',
            f'    Local window size: {self.local_window_size}\n',
            f'    Start pruning stage: {self.start_prune_stage}\n',
            f'    Specific mse layer: {self.specific_mse_layer}\n',
            f'    Beta: {self.beta}'
        )

    def lf_anchor(
        self,
        logits_BlV,
        si,
        mse_difference,
        compress_ratio_,
        local_window_size,
        beta=0.8
    ):
        B = logits_BlV.shape[0]             # logits B=1
        patch_size = self.patch_nums[si]

        # Upsample logits and ΔF to current resolution
        upsampled_logits = torch.nn.functional.interpolate(
            logits_BlV.reshape(B, -1, self.patch_nums[si-1], self.patch_nums[si-1]),    # [B, N_{prev-scale}^2, V] --> [B, V, N_{prev-scale}, N_{prev-scale}]
            size=(patch_size, patch_size), mode='nearest'
        )
        upsampled_mse = torch.nn.functional.interpolate(
            mse_difference.reshape(B, 1, self.patch_nums[si-1], self.patch_nums[si-1]), # [B, N_{prev-scale}^2] --> [B, 1, N_{prev-scale}, N_{prev-scale}]
            size=(patch_size, patch_size), mode='nearest'
        )

        metric = upsampled_mse

        # Select high/low-frequency tokens
            # sort topk (MSE lager ==> high-frequency)
        high_freq_idx = metric.reshape(B, -1).sort(1)[1][:, -int(patch_size * patch_size * compress_ratio_):]
        low_freq_idx = get_reverse_indices(high_freq_idx, patch_size ** 2)

        # Generate candidate anchors
        anchor_candidates = generate_anchors_from_flat(patch_size, local_window_size).to(logits_BlV.device)     # Take the top-left corner for per-window
        anchor_candidates = anchor_candidates.unsqueeze(0).expand(B, -1, -1)    # for cond+uncond branch

        # Match tokens to their best anchors
        best_anchor_map, anchor_similarities = map_to_best_corner(
            patch_size,
            local_window_size,
            torch.arange(0, patch_size ** 2)[None, :].expand(B, -1).to(upsampled_logits.device),
            upsampled_logits.permute(0, 2, 3, 1),
            anchor_candidates
        )
        anchor_candidates = anchor_candidates[..., 0] * patch_size + anchor_candidates[..., 1]

        # Merge anchors with high-frequency tokens
        if B > 1:
            combined_idx = torch.cat([anchor_candidates, high_freq_idx], dim=1)
            min_len = min([i.unique().shape[0] for i in combined_idx])
            high_freq_idx = torch.stack([i.unique(sorted=False)[:min_len] for i in combined_idx])

        # Update low-frequency tokens
        low_freq_idx = get_reverse_indices(high_freq_idx, patch_size ** 2)

        # Mark which low-frequency tokens can be recovered by anchors
        max_sim = anchor_similarities.max(-1)[0].gather(index=low_freq_idx, dim=1)
        recoverable_mask = max_sim > beta
        low_to_anchor_idx = best_anchor_map.gather(dim=1, index=low_freq_idx)

        return high_freq_idx, low_freq_idx, recoverable_mask, low_to_anchor_idx

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        #* ------ sparsevar params ------
        # start_prune_stage=10,
        # compress_method="sparsevar",
        # local_window_size=4,
        # compress_ratio=0.6,
        # specific_mse_layer=3,
        return_features=False,
        # beta=0.99
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule

        #* ------ sparsevar ------
        summed_codeses=[]
        mse_differences=[]
        avg_cossims=[]
        mse_difference=None

        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        kv_compact = self.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)

        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')

        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0
        #* ------ sparsevar ------ 
        self.patch_nums=[i[1] for i in scale_schedule]
        hr_idx=None
        this_stage_image=None

        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            cur_L += np.array(pn).prod()

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                # if need_to_pad:
                #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            layer_idx = 0
            for block_idx, b in enumerate(self.block_chunks):
                # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    #* ------ sparsevar ------
                    if si >= self.start_prune_stage and self.compress_method=="sparsevar":
                        if self.compress_method=="sparsevar":
                            hr_idx, lr_idx, recover_lr, anchor_lr = self.lf_anchor(     # hr_index: high-frequency + anchor (compute), lr_idx: low-frequency (skip)
                                logits_BlV_copy, si, mse_difference, compress_ratio_, self.local_window_size, self.beta)
                        # hr_idx=hr_idx.sort(1)[0]
                        hr_idx = hr_idx.reshape(B, -1)
                        hr_idx_repeat = hr_idx.repeat(2, 1)
                        x_origin = last_stage
                        hr_idx_expand = hr_idx_repeat[..., None].expand(-1, -1, last_stage.shape[-1])
                        last_stage = torch.gather(last_stage, dim=1, index=hr_idx_expand)

                if not self.add_lvl_embeding_only_first_block: 
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)

                origin_x = last_stage
                for m in b.module:
                    if hr_idx is not None:
                        seq_len = pn[1]**2
                    else:
                        seq_len = None
                    last_stage, _ = m(
                        x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None,
                        attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid,
                        scale_ind=si,
                        hr_idx=hr_idx,
                        seq_len=seq_len
                    )
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                        last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1

                if si >= self.start_prune_stage-1:
                    if block_idx == self.specific_mse_layer and self.compress_method == "sparsevar":    # select No.3 chunk to get MSE
                        if hr_idx is not None:
                            x_after = x_origin.to(last_stage)
                            x_after = x_after.scatter(1, hr_idx_expand, last_stage)
                            x_before = x_origin.to(origin_x)
                            x_before = x_before.scatter(1,hr_idx_expand,origin_x)
                        else:
                            x_after = last_stage
                            x_before = origin_x
                        mse_difference = calculate_mse(x_after, x_before)
                        mse_difference = torch.stack([mse_difference[:B], mse_difference[B:]]).mean(0)      # Take the average of the two branches (CFG)
                        if return_features:
                            mse_differences.append(mse_difference)
                        
                        compress_ratio_ = ((mse_difference/mse_difference.max()) > self.compress_ratio).float().mean(1).max()

            if si >= self.start_prune_stage and self.compress_method == "sparsevar":    # 被排除 token 复制最相似 anchor 的预测
                x_origin = x_origin.to(last_stage)
                x_origin.scatter_(1, hr_idx_expand, last_stage)

                source_flag = torch.gather(x_origin, dim=1, index=anchor_lr.unsqueeze(-1).repeat(2, 1, last_stage.shape[-1]))
                dst = torch.gather(x_origin, dim=1, index=lr_idx.unsqueeze(-1).repeat(2,1,last_stage.shape[-1]))
                source_flag = dst*(1-recover_lr.to(dst).repeat(2,1).unsqueeze(-1)) + (recover_lr.to(dst).repeat(2,1).unsqueeze(-1)*source_flag)
                
                x_origin.scatter_(1,lr_idx.unsqueeze(-1).repeat(2,1,last_stage.shape[-1]), source_flag)
                last_stage = x_origin

            if (cfg != 1) and add_cfg_on_logits:
                # print(f'add cfg on add_cfg_on_logits')
                logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            if self.use_bit_label:
                if self.compress_method == "sparsevar":     # 
                    logits_BlV_copy = deepcopy(logits_BlV)
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]

            if vae_type != 0:
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]

                if si >= self.start_prune_stage and (self.compress_method == "sparsevar"):
                    index_temp = torch.zeros(B, pn[1]**2).to(lr_idx.device)
                    index_temp.scatter_(dim=1, index=lr_idx, src=(~recover_lr).to(index_temp))

                    if self.apply_spatial_patchify:
                        # codes 形状: [B, d, 1, H_vae, W_vae]，其中 H_vae/W_vae = 2× transformer 分辨率
                        H_vae, W_vae = codes.shape[-2], codes.shape[-1]
                        # 把 transformer 分辨率上的低频 mask 上采样到 VAE 分辨率
                        index_temp_2d = index_temp.reshape(B, 1, pn[1], pn[2])     # [B,1,N,N]
                        index_temp_2d = torch.nn.functional.interpolate(
                            index_temp_2d, size=(H_vae, W_vae), mode='nearest'
                        )                                                          # [B,1,H_vae,W_vae]
                        index_temp_flat = index_temp_2d.reshape(-1)
                        codes_flat = codes.view(1, codes.shape[1], -1)             # 通道数用 codes.shape[1]，别写死 32
                    else:
                        codes_flat = codes.view(1, 32, -1)
                        index_temp_flat = index_temp.view(-1)

                    codes_flat[:, :, index_temp_flat.bool()] = 0
                    codes=codes_flat.view(codes.shape)
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    summed_codes += codes

                if return_features:
                    summed_codeses.append(deepcopy(summed_codes))
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)

            if si != num_stages_minus_1:
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        # return ret, idx_Bl_list, img, None
        return ret, idx_Bl_list, img
