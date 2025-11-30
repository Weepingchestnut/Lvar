"""
Definitions of blocks of VAR transformer model.
"""

import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import DropPath

# Import flash_attn's attention
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
# not slow attn, for torch >= 2.0.0 and CUDA backend, it support flash attn 2 for fast inference
# https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
# from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
# -->
from torch.nn.functional import scaled_dot_product_attention as torch_attn     # q, k, v: BHLc
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.infinity.basic_infinity import FFN, CrossAttention, FFNSwiGLU, get_dropout_layer


# ? Uncomment this function if you want to benchmark sppedup with vanilla attn.
# def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
#     attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
#     if attn_mask is not None:
#         attn.add_(attn_mask)
#     return (
#         F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
#         if dropout_p > 0
#         else attn.softmax(dim=-1)
#     ) @ value


# Import flash_attn's fused ops
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func
    flash_fused_op_installed = True
except ImportError:
    dropout_add_layer_norm = dropout_add_rms_norm = fused_mlp_func = None
    flash_fused_op_installed = False
    
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight

from matplotlib import pyplot as plt


def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x


def masked_previous_scale_cache(cur_x, num_remain, cur_shape):
    B, L, c = cur_x.shape
    mean_x = cur_x.view(B, cur_shape[1], cur_shape[2], -1).permute(0, 3, 1, 2)
    mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x, (1, 1)).permute(0, 2, 3, 1).view(B, 1, c)      # [bs, 1, C]
    mse_difference = torch.sum((cur_x - mean_x)**2, dim=-1, keepdim=True)
    select_indices = torch.argsort(mse_difference,dim=1, descending=True)
    filted_select_indices=select_indices[:, :num_remain, :]

    def merge(merged_cur_x):
        return torch.gather(merged_cur_x, dim=1, index=filted_select_indices.repeat(1, 1, c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_hw=None):
        unmerged_cache_x_ = unmerged_cache_x.view(B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
        unmerged_cache_x_ = torch.nn.functional.interpolate(unmerged_cache_x_, size=(cur_shape[1], cur_shape[2]), mode='area').permute(0, 2, 3, 1).view(B, L, c)
        unmerged_cache_x_.scatter_(dim=1,index=filted_select_indices.repeat(1,1,c),src=unmerged_cur_x)
        return unmerged_cache_x_

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx


# 1/2 : [... (1, 23, 46), (1, 30, 60), (1, 37, 74), (1, 45, 90), (1, 60, 120)]
# 1.333/1  (1, 36, 27), (1, 48, 36), (1, 60, 45), (1, 72, 54) (1,84,63)
# 2/1:  (1, 46, 23), (1, 60, 30), (1, 74, 37), (1, 90, 45) (1,120,60)
# 1/1 , (13, 32, 32), (15, 40, 40), (17, 48, 48), (21, 64, 64), (1, 84, 84)]
# def compute_merge(x: torch.Tensor, prune_scale_list=[32, 40], is_later_layer=False, x_shape=None) -> Tuple[Callable, ...]:
#     _, original_h, original_w = x_shape
#     original_tokens = original_h * original_w

#     if original_w in prune_scale_list and is_later_layer:
#         ratio_hard_code = {32:0.4, 40:0.5}
#         ratio = ratio_hard_code[original_w]
#         r = int(x.shape[1] * ratio)
#         m, u, id_fn = masked_previous_scale_cache(x, x.shape[1]-r, x_shape)
#     else:
#         m, u, id_fn = (do_nothing, do_nothing, do_nothing)

#     m_a, u_a = (m, u)

#     return m_a, u_a, id_fn  # Okay this is probably not very good


# for w/o skip scales, cache scale-10 (40x40), prune scale-11 -12
def compute_merge(
    x: torch.Tensor, prune_scale_list=None, ratio_hard_code=None,
    is_later_layer=False, x_shape=None
) -> Tuple[Callable, ...]:
    _, original_h, original_w = x_shape
    original_tokens = original_h * original_w

    if original_w in prune_scale_list and is_later_layer:
        # ratio_hard_code = {48:0.6, 64:0.7}
        ratio = ratio_hard_code[original_w]
        r = int(x.shape[1] * ratio)
        m, u, id_fn = masked_previous_scale_cache(x, x.shape[1]-r, x_shape)
    else:
        m, u, id_fn = (do_nothing, do_nothing, do_nothing)

    m_a, u_a = (m, u)

    return m_a, u_a, id_fn  # Okay this is probably not very good


def apply_rotary_emb_fastvar(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier, rope2d_normalized_by_hw, scale_ind, rope_idx_fn=None, seq_len=0):
    qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = np.sum([item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]])
        rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(qk.device)
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, :, :, :, start:start+seq_len] # rope_cache shape: [2, 1, 1, 1, seq_len, half_head_dim]
        # TODO need to add pos gather here
        if rope_idx_fn is not None and rope_idx_fn.__name__ != 'do_nothing':
            rope_idx = rope_idx_fn()    # [bs, pruned_seq_len, 1]
            rope_cache = rope_cache.repeat(1,1,2,1,1,1)
            rope_cache = torch.gather(rope_cache,
                                      index=rope_idx.reshape(1,1,rope_idx.shape[0],1,rope_idx.shape[-2],
                                      rope_idx.shape[-1]).repeat(2,1,1,1,1,rope_cache.shape[-1]), dim=4)
        qk = qk.reshape(*qk.shape[:-1], -1, 2) # (2, batch_size, heads, seq_len, half_head_dim, 2)
        qk = torch.stack([
            rope_cache[0] * qk[...,0] - rope_cache[1] * qk[...,1],
            rope_cache[1] * qk[...,0] + rope_cache[0] * qk[...,1],
        ], dim=-1) # (2, batch_size, heads, seq_len, half_head_dim, 2), here stack + reshape should not be concate
        qk = qk.reshape(*qk.shape[:-2], -1) #(2, batch_size, heads, seq_len, head_dim)
        q, k = qk.unbind(dim=0) # (batch_size, heads, seq_len, head_dim)
    return q, k


class FastVARSelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, rope_idx=None, ori_len=0):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape
        
        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        if self.using_flash: q, k, v = qkv.unbind(dim=2); L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16
        if rope2d_freqs_grid is not None:
            q, k = apply_rotary_emb_fastvar(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind, rope_idx_fn=rope_idx, seq_len=ori_len) #, freqs_cis=freqs_cis)
        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim) # 10,521
        
        if self.using_flash:
            if attn_bias_or_two_vector is not None: # training
                kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
            else:                                   # inference (autoregressive sampling)
                kw = dict()
            oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw).view(B, L, C)
        else:
            # if self.cos_attn: q, k are in fp32; v is in bf16
            # else: q, k, v are in bf16
            if self.use_flex_attn and attn_fn is not None:
                oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            else:
                # --- flashattn ---
                # q, k, v = q.transpose(1,2), k.transpose(1,2),v.transpose(1,2)
                # oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale).reshape(B, L, C)
                # --- torch attn ---
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    oup = torch_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C) #b head l d
            # oup: bf16

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'


class FastVARCrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
        # pruning setting
        cached_scale: int = 8, prune_ratio: tuple = (0.4, 0.5)
    ):
        super(FastVARCrossAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = FastVARSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
            use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        )
        self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        
        self.shared_aln = shared_aln
        if self.shared_aln: # always True
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
        
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1
        
        self.checkpointing_sa_only = checkpointing_sa_only

        # * fastvar add
        self.previous_scale_cache_self_attn = None
        self.previous_scale_cache_cross_attn = None
        self.previous_scale_cache_ffn = None
        
        scales_init = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
        # self.cached_size = [24, 24] # we cahce the scale at 24 as cached feature for subsequant feature restoration
        self.cached_size = [scales_init[cached_scale], scales_init[cached_scale]]
        self.prune_scale_list = scales_init[cached_scale+1:cached_scale+3]
        print(f'\n[Pruning Setting] cache_scale={cached_scale}, scale size is {self.cached_size}')
        if num_heads == 16:
            self.prune_layer_range = range(3,28)
            print(f'    2B model, pruning layers are {list(self.prune_layer_range)}')
        else:
            self.prune_layer_range = range(4,35)
            print(f'    8B model, pruning layers are {list(self.prune_layer_range)}')
        
        assert len(self.prune_scale_list) == len(prune_ratio), f'pruning scale != pruning ratio'
        self.ratio_hard_code = {}
        for i, scale in enumerate(self.prune_scale_list):
            print(f'Scale-{scale} use pruning, ratio is {prune_ratio[i]}')
            self.ratio_hard_code[scale] = prune_ratio[i]

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, layer_idx=-1, x_shape=None):
        gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        is_later_layer = True if layer_idx in list(self.prune_layer_range) else False
        merge_fn, unmerge_fn, idx_fn = compute_merge(x, prune_scale_list=self.prune_scale_list, ratio_hard_code=self.ratio_hard_code,
                                                     is_later_layer=is_later_layer,x_shape=x_shape)
        shortcut = x
        x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=merge_fn(x), scale=scale1, shift=shift1)
        x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind, rope_idx=idx_fn, ori_len=shortcut.shape[1]).mul_(gamma1)

        x_sa = unmerge_fn(x_sa,self.previous_scale_cache_self_attn,self.cached_size)
        if x.shape[1] in [self.cached_size[0]*self.cached_size[1]]:
            self.previous_scale_cache_self_attn = x_sa
        x = shortcut + self.drop_path(x_sa)

        merge_fn, unmerge_fn, idx_fn = compute_merge(x, prune_scale_list=self.prune_scale_list, ratio_hard_code=self.ratio_hard_code,
                                                     is_later_layer=is_later_layer,x_shape=x_shape)
        x_ca = unmerge_fn(self.ca(self.ca_norm(merge_fn(x)), ca_kv).float().mul_(self.ca_gamma),self.previous_scale_cache_cross_attn, self.cached_size)
        if x.shape[1] in [self.cached_size[0]*self.cached_size[1]]:
            self.previous_scale_cache_cross_attn = x_ca
        x = x + x_ca

        merge_fn, unmerge_fn, idx_fn = compute_merge(x, prune_scale_list=self.prune_scale_list, ratio_hard_code=self.ratio_hard_code,
                                                     is_later_layer=is_later_layer,x_shape=x_shape)
        x_ffn = unmerge_fn(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=merge_fn(x), scale=scale2, shift=shift2)).mul(gamma2),self.previous_scale_cache_ffn,self.cached_size)
        if x.shape[1] in [self.cached_size[0] * self.cached_size[1]]:
            self.previous_scale_cache_ffn = x_ffn

        x = x + self.drop_path(x_ffn) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP

        return x

    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'

