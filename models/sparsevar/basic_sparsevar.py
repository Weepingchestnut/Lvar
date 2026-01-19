"""
Definitions of blocks of VAR transformer model.
"""

import math
from functools import partial
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import flash_attn's attention
from flash_attn import flash_attn_func  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc
from timm.layers import DropPath
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as torch_attn  # q, k, v: BHLc
from torch.utils.checkpoint import checkpoint

from models.infinity.basic_infinity import FFN, FFNSwiGLU, get_dropout_layer

# Import flash_attn's fused ops
try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    flash_fused_op_installed = True
except ImportError:
    dropout_add_layer_norm = dropout_add_rms_norm = fused_mlp_func = None
    flash_fused_op_installed = False
    
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


def apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier, rope2d_normalized_by_hw, scale_ind,hr_idx=None,seq_len=None):
    qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        if seq_len is None:
            seq_len = qk.shape[3]
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = np.sum([item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]])
        rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(qk.device)
        assert start+seq_len <= rope2d_freqs_grid[str(tuple(scale_schedule))].shape[4]
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, :, :, :, start:start+seq_len] # rope_cache shape: [2, 1, 1, 1, seq_len, half_head_dim]
        if hr_idx is not None:
            if q.shape[0]>2:
                rope_cache = rope_cache.expand(-1,2,q.shape[0],q.shape[1],-1,-1).gather(index=hr_idx[None,None,:,None,:,None].\
                                                                            repeat(1,1,2,1,1,1).expand(2,2,-1,q.shape[1],-1,rope_cache.shape[-1]),dim=4)
            else:
                rope_cache = rope_cache.gather(index=hr_idx[None,None,None,:,:,None].expand(rope_cache.shape[0],-1,-1,-1,-1,rope_cache.shape[-1]),dim=4)
        qk = qk.reshape(*qk.shape[:-1], -1, 2) #(2, batch_size, heads, seq_len, half_head_dim, 2)
        qk = torch.stack([
            rope_cache[0] * qk[...,0] - rope_cache[1] * qk[...,1],
            rope_cache[1] * qk[...,0] + rope_cache[0] * qk[...,1],
        ], dim=-1) # (2, batch_size, heads, seq_len, half_head_dim, 2), here stack + reshape should not be concate
        qk = qk.reshape(*qk.shape[:-2], -1) #(2, batch_size, heads, seq_len, head_dim)
        q, k = qk.unbind(dim=0) # (batch_size, heads, seq_len, head_dim)
    return q, k


class SelfAttention(nn.Module):
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
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, 
                hr_idx=None, seq_len=None):
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
            q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind,
                                    hr_idx=hr_idx,seq_len=seq_len) #, freqs_cis=freqs_cis)
        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
        
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
                # oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
                # -->
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    oup = (torch_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0)
                           .transpose(1, 2)
                           .reshape(B, L, C))
            # oup: bf16
        prob_importance = compute_query_token_importance_with_key_probes(q, k)
        return self.proj_drop(self.proj(oup)), prob_importance

    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'


def compute_query_token_importance_with_key_probes(query, key, scale=1.0, attn_mask=None, probe_ratio=0.1):
    """
    ZipVL-style: Use a small set of probe keys to estimate query token importance.

    Args:
        query: [B, H, Q, D] - all query tokens
        key:   [B, H, K, D] - full key (may be long KV cache)
        scale: float - usually sqrt(D)
        attn_mask: optional [B, 1, Q, P]
        probe_ratio: fraction of key tokens to use as probes

    Returns:
        importance: [B, Q] - per-query-token importance scores
    """
    B, H, Q, D = query.shape
    K = key.shape[2]
    probe_len = max(1, int(K * probe_ratio))
    device = key.device

    # 1. 选择 probe keys（最近 + 随机）
    probe_recent = torch.arange(K - probe_len, K, device=device)
    if probe_len>1:
        probe_random = torch.randint(0, K - probe_len, (probe_len,), device=device)
        probe_ids = torch.cat([probe_recent, probe_random]).unique().clamp(0, K - 1)  # [P]
    else:
        probe_ids = torch.cat([probe_recent]).unique().clamp(0, K - 1)  # [P]
    probe_key = key[:, :, probe_ids, :]  # [B, H, P, D]

    # 2. attention logits: [B, H, Q, P]
    attn_logits = torch.matmul(query, probe_key.transpose(-2, -1)) / scale
    if attn_mask is not None:
        attn_logits += attn_mask[:, :, :, probe_ids]  # masking selected probe positions

    attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, Q, P]

    # 3. importance = 每个 query token 在 probe 上的最大 attention
    attn_max = attn_weights.max(dim=-1).values  # [B, H, Q]
    importance = attn_max.mean(dim=1)  # average over heads → [B, Q]

    return importance


class CrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False,
    ):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        cos_attn = False    # TODO: never use cos attn in cross attention with T5 kv
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)
        
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
    
    def forward(self, q, ca_kv):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        
        kv_compact = F.linear(kv_compact, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc
        # attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens
        
        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        
        if self.cos_attn:   # always False
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)
        
        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        
        cu_seqlens_q = torch.arange(0, Lq * (B+1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:    # todo: fp16 or bf16?
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
        
        prob_importance = compute_query_token_importance_with_key_probes(
            q_compact.reshape(B,q_compact.shape[0]//B,q_compact.shape[1],-1).permute(0,2,1,3), 
            kv_compact.permute(1,2,0,3))
        return self.proj_drop(self.proj(oup)),prob_importance
    
    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, cos_attn={self.cos_attn}'


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
    ):
        super(CrossAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = SelfAttention(
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
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, 
                hr_idx=None, seq_len=None):    # todo: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm
        with torch.autocast('cuda', enabled=False):    # disable half precision
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        if self.fused_norm_func is None:
            x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid)
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        else:
            x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                x_sa, prob_importance = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind, 
                                                hr_idx=hr_idx, seq_len=seq_len)
            x = x + self.drop_path(x_sa.mul_(gamma1))
            out, _ = self.ca(self.ca_norm(x), ca_kv)
            x = x + out.float().mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        return x, prob_importance
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'


def main():
    dev = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = torch.Generator(device=dev)
    # for Li in ([1, 3, 5], [1, 3]):
    rng.manual_seed(0)
    B, H, cq, ckv = 4, 8, 64, 96
    Cq = H*cq
    Ckv = H*ckv
    
    Li = [5, 4, 7, 6]
    Lq = 10
    L = max(Li)
    attn_bias = torch.zeros(B, 1, Lq, L, device=dev)
    for i, x in enumerate(Li):
        attn_bias[i, 0, :, x:] = -torch.inf
    
    q = torch.randn(B, Lq, H, cq, generator=rng, device=dev)
    k = torch.randn(B, L, H, ckv, generator=rng, device=dev)
    v = torch.randn(B, L, H, ckv, generator=rng, device=dev)
    tq, tk, tv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)    # BHLc
    
    seqlen_k = torch.tensor(Li, dtype=torch.int32, device=dev)
    cu_seqlens_k = F.pad(torch.cumsum(seqlen_k, dim=0, dtype=torch.torch.int32), (1, 0))
    kv = torch.stack([k, v], dim=2)
    kv_compact = torch.cat([kv[i, :Li[i]] for i in range(B)], dim=0)
    
    ca = CrossAttention(for_attn_pool=False, embed_dim=Cq, kv_dim=Ckv, num_heads=H)
    CrossAttention.forward
    ca(q, (kv_compact, cu_seqlens_k, max(Li))).mean().backward()


if __name__ == '__main__':
    main()
