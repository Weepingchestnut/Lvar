import math
import sys
debugger_attached = hasattr(sys, 'gettrace') and sys.gettrace() is not None
import numpy as np
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
from torch.utils.checkpoint import checkpoint
from timm.layers import DropPath, drop_path

# Import flash_attn's attention
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc

from models.infinity.basic_infinity import FFN, CrossAttention, FFNSwiGLU, apply_rotary_emb, get_dropout_layer, precompute_rope2d_freqs_grid
from models.sparvar.sparse_attn import SparseDiffAttn
from models.sparvar.sparse_attn_layer_counter import LayerCounter

# not slow attn, for torch 2.6.0, it's flash attn 2 
# https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc

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


class SparseFlexAttn(nn.Module):
    def __init__(self, scale_schedule, kernel_schedule, q_scale_idx, attn_sink_scale, block_size=_DEFAULT_SPARSE_BLOCK_SIZE, num_heads: int = 16):
        super().__init__()

        S_q = scale_schedule[q_scale_idx][0] * scale_schedule[q_scale_idx][1]
        S_kv = sum([h * w for h, w in scale_schedule[:q_scale_idx + 1]])
        self.block_size = block_size

        self.flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

        self.mask_mod = self.create_cross_scale_block_attn_mask(
            scale_schedule=scale_schedule,
            kernel_schedule=kernel_schedule,
            q_scale_idx=q_scale_idx,
            attn_sink_scale=attn_sink_scale
        )

        self.block_mask = create_block_mask(
            self.mask_mod,
            B=1,
            H=num_heads,
            Q_LEN=S_q,
            KV_LEN=S_kv,
            device='cuda',
            BLOCK_SIZE=block_size
        )

    def create_cross_scale_block_attn_mask(
        self,
        scale_schedule=None,
        kernel_schedule=None,
        q_scale_idx: int = 12,
        attn_sink_scale: int = 5,
        device: str = "cuda",
    ):
        """
        一个统一的稀疏 attention mask mod，用于为Infinity模型的完整注意力矩阵
        （例如 4096x10521）进行局部稀疏化。

        该函数能够根据 Key/Value 的一维索引 kv_idx，自动推断其所属的历史尺度、
        该尺度的网格尺寸，以及在该尺度内的局部坐标，然后应用局部注意力逻辑。
        """

        H_q, W_q = scale_schedule[q_scale_idx]
        S_q = H_q * W_q

        # 计算所有历史 Key/Value 尺度的 token 数量和边界
        relevant_schedule = scale_schedule[:q_scale_idx + 1]
        relevant_kernel_schedule = kernel_schedule[:q_scale_idx + 1]

        kv_tokens_per_scale = [h * w for h, w in relevant_schedule]
        kv_cumulative_tokens = np.cumsum(kv_tokens_per_scale)
        kv_slice_indices = np.insert(kv_cumulative_tokens, 0, 0)
        S_k = kv_cumulative_tokens[-1]

        # Calculate the token boundary for the dense attention sink
        attn_sink_boundary = int(kv_slice_indices[attn_sink_scale]) if attn_sink_scale > 0 else 0

        scale_schedule_tensor = torch.tensor(relevant_schedule, device=device, dtype=torch.int)
        kv_slice_indices_tensor = torch.tensor(kv_slice_indices, device=device, dtype=torch.int)
        kernel_schedule_tensor = torch.tensor(relevant_kernel_schedule, device=device, dtype=torch.int)

        def gen_fine_grained_mask_mod(b, h, q_idx, kv_idx):

            # Condition 1: Is the Key/Value token in the dense "attention sink" region?
            # If so, it's always attended to.
            attn_sink_mask = (kv_idx < attn_sink_boundary)

            # --- The rest of the logic is for the sparse region ---
            # 1. 计算 Query 的 2D 坐标 (这部分不变)
            q_y, q_x = q_idx // W_q, q_idx % W_q
            
            # 2. 核心逻辑: 确定每个 kv_idx 属于哪个历史尺度
            #    torch.searchsorted 是一个高效的查找操作，它会返回每个 kv_idx
            #    应该插入到 slice_indices 中的哪个位置，从而确定其所属的尺度。
            #    我们减去1来获得正确的0-based尺度索引。
            scale_index_k = torch.searchsorted(kv_slice_indices_tensor, kv_idx, right=True) - 1
            
            # 3. 根据尺度索引，获取该尺度的网格尺寸 (H_k, W_k)
            #    使用 torch.gather 来避免在 vmap 环境中的索引问题
            H_k_tensor = torch.gather(scale_schedule_tensor[:, 0], 0, scale_index_k)
            W_k_tensor = torch.gather(scale_schedule_tensor[:, 1], 0, scale_index_k)
            
            # 4. 计算 kv_idx 在其自身尺度内的局部一维索引
            start_indices_k = torch.gather(kv_slice_indices_tensor, 0, scale_index_k)
            kernel_size_k = torch.gather(kernel_schedule_tensor, 0, scale_index_k)
            relative_kv_idx = kv_idx - start_indices_k
            
            # 5. 计算 Key/Value 的 2D 坐标 Calculate Key/Value's 2D coordinates
            kv_y, kv_x = relative_kv_idx // W_k_tensor, relative_kv_idx % W_k_tensor
            
            # 6. 坐标重缩放，找到 Query 在 Key 网格中的对应中心点 Rescale coordinates
            center_y = torch.round((q_y.float() / H_q) * H_k_tensor)
            center_x = torch.round((q_x.float() / W_q) * W_k_tensor)
            
            # 7. 判断 Key/Value token 是否在中心点 KERNEL_SIZE//2 的邻域内
                # Check neighborhood using the per-scale kernel size
            half_kernel = kernel_size_k // 2
            vertical_mask = (center_y - kv_y).abs() <= half_kernel
            horizontal_mask = (center_x - kv_x).abs() <= half_kernel
            is_in_sparse_region = vertical_mask & horizontal_mask
            
            # return vertical_mask & horizontal_mask
            # A token is attended to if it's in the dense region OR in the local sparse region.
            return attn_sink_mask | is_in_sparse_region
    
        # print("Step 1: Generating internal fine-grained mask...")
        fine_grained_mask = create_mask(gen_fine_grained_mask_mod, 1, 1, S_q, S_k, device=device).squeeze()

        # --- Step 2: Convert to a Pure Block-Sparse Mask ---
        # print(f"Step 2: Converting to {block_size}x{block_size} block-sparse format...")
        # Pad the fine-grained mask to be divisible by block_size
        pad_q = (self.block_size - S_q % self.block_size) % self.block_size
        pad_k = (self.block_size - S_k % self.block_size) % self.block_size
        padded_mask = F.pad(fine_grained_mask, (0, pad_k, 0, pad_q), "constant", False)

        padded_S_q, padded_S_k = padded_mask.shape
        num_q_blocks = padded_S_q // self.block_size
        num_k_blocks = padded_S_k // self.block_size

        # Reshape, permute, and reduce to get the block-level mask
        # The logic is: if ANY value in a block is True, the whole block becomes True.
        block_level_mask = (
            padded_mask.reshape(num_q_blocks, self.block_size, num_k_blocks, self.block_size)
            .permute(0, 2, 1, 3)
            .any(dim=(-1, -2))
        )

        # --- Step 3: Return a Simple Lookup Function ---
        # print("Step 3: Returning simplified block-lookup mask function.")
        
        # This closure captures the pre-computed block_level_mask
        def final_block_mask_mod(b, h, q_idx, kv_idx):
            q_block_idx = q_idx // self.block_size
            k_block_idx = kv_idx // self.block_size
            # This is now a simple, fast lookup
            return block_level_mask[q_block_idx, k_block_idx]

        return final_block_mask_mod
    
    def forward(self, q, k, v, scale=None):
        oup = self.flex_attention(
            q.to(v.dtype), k.to(v.dtype), v, 
            block_mask=self.block_mask, 
            scale=scale
        )

        return oup


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0,
        attn_config=None,
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
            # ---- 新增：缓存与状态 ----
            self.register_buffer("_scale_cache", None, persistent=False)   # 运行时缓存（fp32）
            self._scale_cache_valid = False
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

        #* ------ CS4A + CSLA ------
        # attn_config = GLOBAL_CONFIG['attn']
        # print('f    In SelfAttention:')
        # print(f'{attn_config=}')
        spsd_scale = attn_config['decision_scale']
        self.bound_layer = attn_config['bound_layer']
        print(f'--- [boundary layer]: {self.bound_layer} ---')
        
        layer_num, layer_counter = LayerCounter.build_for_layer(is_attn_sparse=True)
        self.sparse_attn = SparseDiffAttn(
            layer_num, layer_counter, use_o_cache=True, spsd_scale=spsd_scale, 
            # wind_size=[5, 7]
        )
        self.spsd_scale = spsd_scale
    
    # ---- 新增：在切换 train/eval、加载权重后自动失效缓存 ----
    def train(self, mode: bool = True):
        self._scale_cache_valid = False
        return super().train(mode)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self._scale_cache_valid = False
        return super().load_state_dict(state_dict, strict=strict)
    
    @torch.no_grad()
    def _update_scale_cache_if_needed(self, device, dtype=torch.float32):
        """
        仅在 eval() 模式下使用：把 scale_mul 的 exp 结果缓存起来，避免每次 forward 都计算。
        - device: 与当前 q/k 的 device 对齐，避免跨设备拷贝
        - dtype: 保持为 fp32 即可（后续 q/k 在本模块中已转为 fp32 参与归一化）
        """
        if not self._scale_cache_valid or self._scale_cache is None or self._scale_cache.device != device:
            # clamp + exp 的一次性开销下沉到这里
            cache = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            # 放到与输入一致的 device，保持 fp32（数值更稳）
            self._scale_cache = cache.to(device=device, dtype=dtype, non_blocking=True)
            self._scale_cache_valid = True
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]],
                attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, layer_ind=0, spattn_fn=None):
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
        qkv = (F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
               .view(B, L, 3, self.num_heads, self.head_dim))  # BL3Hc
        if self.using_flash: q, k, v = qkv.unbind(dim=2); L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
        if self.cos_attn:   # always True
            if not self.training:
                self._update_scale_cache_if_needed(device=q.device, dtype=torch.float32)
                scale_mul = self._scale_cache
            else:
                scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            # print(f'after F.normalize: {q.dtype=}')
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            # print(f'after F.normalize: {k.dtype=}')
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16
        if rope2d_freqs_grid is not None:
            q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind) #, freqs_cis=freqs_cis)

        # if B == 1 and self.cached_k.shape[0] == 2:
        if self.cached_k is not None and 2*B == self.cached_k.shape[0]:
            self.cached_k = self.cached_k[:B]; self.cached_v = self.cached_v[:B]
        
        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
        # print(f'\n[scale-{scale_ind}_layer-{layer_ind}] self-attn caching: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')

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
            #* --- for Local sparse ---
            elif spattn_fn is not None and layer_ind > self.bound_layer:
                # print('     CSLA')
                oup = spattn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            #* --- for TopK sparse ---
            elif scale_ind == self.spsd_scale and layer_ind <= self.bound_layer:
                # print('     CS4A decision scale')
                oup = self.sparse_attn(q.to(v.dtype), k.to(v.dtype), v, scale=self.scale, do_full_step=True, 
                                       scale_ind=scale_ind, layer_ind=layer_ind).transpose(1, 2).reshape(B, L, C)
            elif scale_ind > self.spsd_scale and layer_ind <= self.bound_layer:
                # print('     CS4A sparse scale')
                oup = self.sparse_attn(q.to(v.dtype), k.to(v.dtype), v, scale=self.scale, 
                                       scale_ind=scale_ind, layer_ind=layer_ind).transpose(1, 2).reshape(B, L, C)
            else:
                # print('     o, _ = torch.ops.chipmunk.dense_attn(q.to(v.dtype), k.to(v.dtype), v)')
                # oup = (slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0)
                #        .transpose(1, 2)
                #        .reshape(B, L, C))
                
                # --- CUDA attn ---
                o, _ = torch.ops.chipmunk.dense_attn(q.to(v.dtype), k.to(v.dtype), v)
                # assert lse.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
                oup = o.transpose(1, 2).reshape(B, L, C)
            # oup: bf16
        
        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
        attn_config=None
    ):
        super(CrossAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
            use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            attn_config=attn_config,
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
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, 
                scale_ind=0, layer_ind=0, spattn_fn=None):
        with torch.autocast('cuda', enabled=False):     # disable half precision
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        if self.fused_norm_func is None:
            x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind)      # * add scale_ind
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        else:
            x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, 
                               scale_ind=scale_ind, layer_ind=layer_ind, spattn_fn=spattn_fn)
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'


if __name__ == '__main__':
    embed_dim = 2048
    num_heads = 2048 // 128
    batch_size = 2
    seq_len = 2304
    pad_to_multiplier = 128
    rope2d_normalized_by_hw = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attn = SelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        proj_drop=0.0,
        tau=1.0,
        cos_attn=True,
        customized_flash_attn=False,
        use_flex_attn=False,
        batch_size=batch_size,
        pad_to_multiplier=pad_to_multiplier,
        rope2d_normalized_by_hw=rope2d_normalized_by_hw
    ).to(device)

    x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32, device=device)

    from utils.dynamic_resolution import dynamic_resolution_h_w
    scale_schedule = [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 6, 6), (1, 8, 8), (1, 12, 12), 
                      (1, 16, 16), (1, 20, 20), (1, 24, 24), (1, 32, 32), (1, 40, 40), 
                      (1, 48, 48), (1, 64, 64)]
    rope2d_freqs_grid = precompute_rope2d_freqs_grid(
        dim=embed_dim // num_heads,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
        pad_to_multiplier=pad_to_multiplier,
        rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        device=x.device
    )

    with torch.no_grad():
        out = attn(
            x,
            attn_bias_or_two_vector=None,
            attn_fn=None,
            scale_schedule=scale_schedule,
            rope2d_freqs_grid=rope2d_freqs_grid,
            scale_ind=0
        )
    print(f"SelfAttention output shape: {out.shape}")
