import copy
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import triton
import yaml
from einops import rearrange
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# from models.sparsevar.sparse_attn_config import GLOBAL_CONFIG, get_kernel_config_attn
from kernels.sparse_attn.sparse_attn_config import GLOBAL_CONFIG, get_kernel_config_attn
from models.sparvar.sparse_attn_layer_counter import LayerCounter
# from models.sparsevar.sparse_attn_ops import (
#     bitpack, bitunpack, 
#     csp_attn, dense_attn, 
#     dense_colsum_attn_q, dense_colsum_attn_s, 
#     mask_to_indices, 
#     naive_dense_colsum_attn, naive_dense_colsum_attn2
# )
from kernels.sparse_attn.sparse_attn_ops import dense_colsum_attn_q, dense_colsum_attn_s, csp_attn


# Determines how many layer "slots" we keep in GPU memory simultaneously
PIPELINE_DEPTH = 2
assert PIPELINE_DEPTH > 1, "Pipeline depth must be greater than 1 - if pipeline depth is 1, this means we are using naive offloading per-layer which doesn't make sense!"

# We create two dedicated streams for all offloading (rather than using per-layer or per-
# object streams) in order to avoid using PyTorch’s internal stream pool (default ~32 streams). 
# That pool can inadvertently cause collisions with the main compute stream every 32nd stream
# instantiation, making memory transfers slow.
global_offload_stream = torch.cuda.Stream()
global_load_stream = torch.cuda.Stream()

# A global dictionary to hold GPU tensors in a round-robin pipeline:
#  gpu_tensors[name] -> [gpu_tensor_for_slot_0, gpu_tensor_for_slot_1, ...]
gpu_tensors = {}


class MaybeOffloadedTensor:
    """
    MaybeOffloadedTensor implements a mechanism for maintaining a sliding pipeline of GPU tensors,
    that are dynamically loaded in and out of pinned CPU memory as the model progresses through layers.
    The pipeline depth is determined by a constant (`PIPELINE_DEPTH`), and the GPU storage is shared
    between layers.
    
    Design Goals:
      1) Maintain a small pipeline of loaded layers (determined by PIPELINE_DEPTH).
      2) Use pinned CPU memory for faster async transfers.
      3) Use a single dedicated CUDA stream to handle CPU↔GPU copies without
         competing with the default compute stream.
      4) Defer GPU allocations until load-time for memory efficiency.
    
    Usage Overview:
      - Instantiate this class with a unique `name` and a `layer_num`.
      - Offload any current GPU tensor using `offload()`.
      - Load back into GPU memory on demand via `load()`.
      - If you need to block until copies are done, call `wait_for_completion()`.
    """

    # Default buffer sizes for pinned CPU memory, tuned for typical shape sizes
    LARGE_BUF_SIZE  = int(1 * 32 * 150000 * 128 * torch.finfo(torch.bfloat16).bits // 8)
    MEDIUM_BUF_SIZE = int(1 * 32 * 90000 * 128 * torch.finfo(torch.bfloat16).bits // 8)
    SMALL_BUF_SIZE  = 1 * 32 * 15000 * 128 * torch.finfo(torch.bfloat16).bits // 8

    @torch.compiler.disable # torch.compile fails to allocate pinned CPU memory :(
    def __init__(self, name: str, layer_num: int, dtype: torch.dtype,
                 device: torch.device, cpu_buf_size: int = LARGE_BUF_SIZE):
        """
        :param name: Unique identifier for this tensor group, shared between layers.
        :param layer_num: Numeric index for the layer; used with PIPELINE_DEPTH
                          to map to a particular slot.
        :param dtype: PyTorch data type, e.g. torch.bfloat16.
        :param device: Device on which GPU allocation will occur (e.g. 'cuda').
        :param cpu_buf_size: Size of the pinned CPU buffer for offloading.
        """
        is_offload_enabled = GLOBAL_CONFIG['offloading']
        if name not in is_offload_enabled:
            raise ValueError(f"Invalid tensor name: {name}. Expected one of: {is_offload_enabled.keys()}")
        self.name = name
        self.layer_num = layer_num
        self.is_offload_enabled = not GLOBAL_CONFIG['offloading']['global_disable_offloading'] and is_offload_enabled[name]
        assert not (self.is_offload_enabled == True and name == 'attn.lse_constants'), "LSE constants cannot be offloaded (i) in Triton because they are passed in as a tuple. You will need to implement this manually yourself; and (ii) in CUDA because they are padded to 16-byte TMA-aligned tensors, and offloading with non-contiguous tensors is not yet tested."
        assert not (self.is_offload_enabled == True and name == 'attn.indices' and GLOBAL_CONFIG['attn']['provider'] == 'cuda' and GLOBAL_CONFIG['attn']['should_compress_indices'] == False), "Non-compressed indices cannot be offloaded in CUDA because they are padded to 16-byte TMA-aligned tensors, and offloading with non-contiguous tensors is not yet tested."
        # Choose a pipeline slot for this layer using modulo:
        self.layer_key = layer_num % PIPELINE_DEPTH
        self.device = device
        self.offload_stream = global_offload_stream
        self.load_stream = global_load_stream
        # Pre-allocate pinned CPU buffer to hold the tensor data
        if self.is_offload_enabled:
            print(f"Offloaded tensor {name} allocated {cpu_buf_size} bytes of pinned CPU memory for {name} layer {layer_num}")
            self.cpu_buf = [torch.empty(cpu_buf_size, dtype=dtype, device="cpu", pin_memory=True) for _ in range(GLOBAL_CONFIG['num_model_invocations_per_inference_step'])]
        else:
            self.gpu_tensor = [None for _ in range(GLOBAL_CONFIG['num_model_invocations_per_inference_step'])]
        # Will store the original shape of the tensor so we can reload properly
        self.real_shape = [None for _ in range(GLOBAL_CONFIG['num_model_invocations_per_inference_step'])]
        self.real_stride = [None for _ in range(GLOBAL_CONFIG['num_model_invocations_per_inference_step'])]
        self.load_completed_event = None

        self.model_invocation_count = 0
        
        # Allocate PIPELINE_DEPTH slots for this tensor name
        if name not in gpu_tensors:
            gpu_tensors[name] = [None] * PIPELINE_DEPTH

    def complete_cur_layer(self):
        self.model_invocation_count += 1

    def get_cur_model_invocation_key(self):
        return self.model_invocation_count % GLOBAL_CONFIG['num_model_invocations_per_inference_step']

    @torch.compiler.disable # disable torch.compile so that we can use pinned memory and .record_stream()
    def offload(self, gpu_tensor: torch.Tensor):
        """
        Asynchronously copy a GPU tensor into this object's pinned CPU buffer on self.offload_stream.
        We remember the shape to reconstruct the tensor when loading back to GPU.

        :param gpu_tensor: Tensor on GPU that will be copied out to CPU memory.
        """
        if not self.is_offload_enabled:
            self.gpu_tensor[self.get_cur_model_invocation_key()] = gpu_tensor
            return
        # Validate that our pinned buffer is large enough
        assert gpu_tensor.numel() <= self.cpu_buf[self.get_cur_model_invocation_key()].numel(), (
            f"Tensor {self.name} is too large to offload - try adjusting MaybeOffloadedTensor.LARGE_BUF_SIZE (requested {gpu_tensor.numel()} elements, available {self.cpu_buf[self.get_cur_model_invocation_key()].numel()} elements)"
        )
        # Record the original shape so we can create a matching GPU tensor on load
        self.real_shape[self.get_cur_model_invocation_key()] = gpu_tensor.size()
        self.real_stride[self.get_cur_model_invocation_key()] = gpu_tensor.stride()          # <── store the stride
        # Perform copy on our dedicated self.offload_stream
        self.offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.offload_stream):
            self.cpu_buf[self.get_cur_model_invocation_key()][:gpu_tensor.numel()].view(gpu_tensor.shape).copy_(gpu_tensor, non_blocking=True)
            gpu_tensor.record_stream(self.offload_stream)

    def offload_cur_value(self):
        """
        Utility function to offload the currently loaded GPU value for this layer.
        If the GPU slot for this layer has data (i.e. was loaded), offload it to CPU.
        """
        self.offload(self.get_loaded_value())

    def get_loaded_value(self):
        """
        Retrieve the GPU tensor that corresponds to this layer_num's slot.
        :return: The GPU tensor if loaded, otherwise raises AssertionError.
        """
        if not self.is_offload_enabled:
            return self.gpu_tensor[self.get_cur_model_invocation_key()]
        
        gpu_tensor = gpu_tensors[self.name][self.layer_key]
        assert gpu_tensor is not None, (
            f"Tensor {self.name} is not loaded yet for layer {self.layer_num}. Please call load_async() first (followed by load_async_wait())"
        )
        return gpu_tensor

    @torch.compiler.disable # disable torch.compile so that we can use pinned memory and tensor.record_stream()
    def load_async(self):
        """
        Load the tensor from this object's pinned CPU buffer back into GPU memory.
        If the GPU slot is not allocated yet, allocate it now. Then copy asynchronously.

        :return: The GPU tensor now loaded into the correct slot.
        """
        key  = self.get_cur_model_invocation_key()
        size = self.real_shape[key]
        if size is None:           # nothing has been off-loaded yet
            return None

        if not self.is_offload_enabled:
            return self.gpu_tensor[key]

        stride = self.real_stride[key]

        # (re)allocate the GPU slot **with identical strides**
        slot = gpu_tensors[self.name]
        need_new = (
            slot[self.layer_key] is None
            or slot[self.layer_key].shape  != size
            or slot[self.layer_key].stride() != stride
        )
        if need_new:
            slot[self.layer_key] = torch.empty_strided(   # <── preserves layout
                size, stride,
                dtype=self.cpu_buf[key].dtype,
                device=self.device,
            )

        gpu_view = slot[self.layer_key]      # this is already non-contiguous if stride says so

        # copy row-major CPU view → strided GPU view
        flat_src = self.cpu_buf[key][:gpu_view.numel()]
        self.load_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.load_stream):
            gpu_view.copy_(flat_src.view(size), non_blocking=True)
            gpu_view.record_stream(self.offload_stream)

        return gpu_view
    
    def load_async_wait(self):
        """
        Instruct the current (default) stream to wait for the self.offload_stream.
        This ensures all CPU↔GPU transfers on self.offload_stream are finished before
        proceeding with compute on the default stream.
        """
        if not self.is_offload_enabled:
            return

        torch.cuda.current_stream().wait_stream(self.load_stream)
        torch.cuda.current_stream().wait_stream(self.offload_stream)

        if self.load_completed_event is not None:
            self.load_completed_event.wait()
            self.load_completed_event = None


class AttnStorage:
    def __init__(self, layer_num: int, init_names: list[str]=[]):
        self.layer_num = layer_num

        self.indices = None
        self.counts = None
        self.out_cache = None
        self.lse_constants = None

        if GLOBAL_CONFIG['offloading']['global_disable_offloading']:
            return

        # for name in init_names:
        if 'out_cache' in init_names:
            self.out_cache = MaybeOffloadedTensor(
                f'attn.out_cache',
                self.layer_num, torch.bfloat16,
                torch.device('cuda'),
                cpu_buf_size=MaybeOffloadedTensor.LARGE_BUF_SIZE
            )
        if 'indices' in init_names:
            self.indices = MaybeOffloadedTensor(
                f'attn.indices',
                self.layer_num, torch.uint8,
                torch.device('cuda'),
                cpu_buf_size=MaybeOffloadedTensor.MEDIUM_BUF_SIZE
            )

    def complete_cur_layer(self):
        if self.indices is not None:
            self.indices.complete_cur_layer()
        if self.counts is not None:
            self.counts.complete_cur_layer()
        if self.out_cache is not None:
            self.out_cache.complete_cur_layer()
        if self.lse_constants is not None:
            self.lse_constants.complete_cur_layer()


    def get_indices(self):
        if self.indices is None:
            return None
        return self.indices.get_loaded_value()

    def set_indices(self, indices: Tensor):
        if self.indices is None:
            self.indices = MaybeOffloadedTensor('attn.indices', self.layer_num, indices.dtype, indices.device)
        self.indices.offload(indices)

    def get_counts(self):
        if self.counts is None:
            return None
        return self.counts.get_loaded_value()

    def set_counts(self, counts: Tensor):
        if self.counts is None:
            self.counts = MaybeOffloadedTensor('attn.counts', self.layer_num, counts.dtype, counts.device)
        self.counts.offload(counts)

    def get_out_cache(self):
        if self.out_cache is None:
            return None
        return self.out_cache.get_loaded_value()

    def set_out_cache(self, out_cache: Tensor):
        if self.out_cache is None:
            self.out_cache = MaybeOffloadedTensor('attn.out_cache', self.layer_num, out_cache.dtype, out_cache.device)
        self.out_cache.offload(out_cache)

    def get_lse_constants(self):
        if self.lse_constants is None:
            return None
        return self.lse_constants.get_loaded_value()
    
    def set_lse_constants(self, lse_constants: Tensor):
        if self.lse_constants is None:
            tensor = lse_constants[0] if isinstance(lse_constants, tuple) else lse_constants
            self.lse_constants = MaybeOffloadedTensor('attn.lse_constants', self.layer_num, tensor.dtype, tensor.device)
        self.lse_constants.offload(lse_constants)

    def load_async(self):
        if self.indices is not None:
            self.indices.load_async()
        if self.counts is not None:
            self.counts.load_async()
        if self.out_cache is not None:
            self.out_cache.load_async()
        if self.lse_constants is not None:
            self.lse_constants.load_async()

    def load_async_wait(self):
        if self.indices is not None:
            self.indices.load_async_wait()
        if self.counts is not None:
            self.counts.load_async_wait()
        if self.out_cache is not None:
            self.out_cache.load_async_wait()
        if self.lse_constants is not None:
            self.lse_constants.load_async_wait()


def upsample_Ocache(
    O_cache: torch.Tensor,      # [B, H, L_S, Dh]
    L_src: int, L_tgt: int,
    # hS: int, wS: int,
    # hK: int, wK: int,
    mode: str = "2d",          # "1d" 或 "2d"
    upsample_mode: str = 'bilinear'      # 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
) -> torch.Tensor:
    
    # assert L_S == hS * wS and L_K == hK * wK, "mismatch"

    B, H, _, Dh = O_cache.shape

    if mode == '1d':
        # 最近邻 1D 重采样（长度 → 新长度）
        idx = torch.round(torch.arange(L_tgt, device=O_cache.device, dtype=torch.float32) * (L_src - 1) / max(L_tgt - 1, 1)).to(torch.long)
        return O_cache.index_select(dim=2, index=idx)
    elif mode == "2d":
        # x = O_cache.permute(0, 1, 3, 2).contiguous()            # [B,H,L,D] -> [B,H,D,L]
        # x = x.view(B, H, Dh, int(L_S**0.5), int(L_S**0.5))                            # [B,H,Dh,hS,wS]
        x = O_cache.view(B, H, int(L_src**0.5), int(L_src**0.5), Dh).permute(0, 1, 4, 2, 3)
        x = x.reshape(B * H, Dh, int(L_src**0.5), int(L_src**0.5))

        if upsample_mode == 'bilinear':
            x = F.interpolate(
                x, size=(int(L_tgt**0.5), int(L_tgt**0.5)), 
                mode=upsample_mode,
                align_corners=False)
        else:
            x = F.interpolate(
                x, size=(int(L_tgt**0.5), int(L_tgt**0.5)), 
                mode=upsample_mode)

        # x = x.view(B, H, Dh, L_K).permute(0, 1, 3, 2).contiguous()  # [B,H,L_K,Dh]
        x = x.view(B, H, Dh, int(L_tgt**0.5), int(L_tgt**0.5))
        x = x.permute(0, 1, 3, 4, 2)
        return x.reshape(B, H, -1, Dh)      # view also True
    else:
        raise ValueError(f"unknown mode={mode}")


@torch.no_grad()
def map_inds_counts_across_scales(
    inds_S: torch.Tensor,          # [B, H, Gs, Ks]  基准尺度S的列索引（已对齐）
    counts_S: torch.Tensor,        # [B, H, Gs]     基准尺度S每组count（通常常数&对齐）
    cur_q_len: int,                      # 目标尺度K的 Q 长度（用于确定 Gk）
    cur_k_len: int,               # 目标尺度下历史KV的总长度（上界）
    bm: int = 192,                  # query-group 大小（64 / 192）
    indices_pad_to: int = 1,      # indices 对齐粒度（通常=64）
    add_sink: bool = False,         # 是否添加 attention sink
    sink_len: int = 0,             # sink 采用 [0:sink_len)，建议设为“前若干尺度 token 总数”
    sink_quota_frac: float = 0.2,  # sink 配额在 indices_count 中占比（仅当 add_sink=True 生效）
    device: torch.device = None,
):
    """
    将 (inds_S, counts_S) 从尺度S映射到cur scale，返回:
      inds_K   [B, H, Gk, Kk]
      counts_K [B, H, Gk]      （各组均为同一个 indices_count，且是 indices_pad_to 的倍数）

    设计要点：
    1) 组映射：Gk 组 -> Gs 组，使用中心点最近邻 (u=(g+0.5)/G * Gs - 0.5) 舍入并 clamp。
    2) 若 add_sink=True：优先写入长度为 sink_quota 的 [0:sink_len)，剩余配额由模板 inds_S 填充（去除与 sink 的重叠）。
    3) 模板填充：用掩码 templ>=sink_len，配合 "很大值屏蔽 + topk(取最小)"，一次性选出需要的列；若仍不足则循环补齐。
    4) 保证所有索引在 [0, K_hist_len)。
    """
    assert inds_S.ndim == 4 and counts_S.ndim == 3, f"{inds_S.shape=}, {counts_S.shape=}"
    B, H, Gs, Ks = inds_S.shape
    if device is None:
        device = inds_S.device
    
    # 计算 Gs/Gk 并断言
    Gk = (cur_q_len + bm - 1) // bm
    # assert Gs == Gs_expect, f"Gs({Gs}) != ceil(L_S/bm)({Gs_expect})"

    # indices_count（对齐为 indices_pad_to 的倍数），假设各组相同（Chipmunk约定）
    # 取 (0,0,0) 位置；如工程里确实 per-(B,H) 不同，可改成 per-(B,H) 广播。
    raw_cnt = int(counts_S[0, 0, 0].item())
    indices_count = max(indices_pad_to, (raw_cnt + indices_pad_to - 1) // indices_pad_to * indices_pad_to)

    # 组映射：Gk 中每组对应到 Gs 的哪一组（1D 索引），然后对 dim=2 做 index_select
    gk_centers = torch.arange(Gk, device=device, dtype=torch.float32) + 0.5
    gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).to(torch.int64).clamp_(0, Gs - 1)  # [Gk]
    templ = inds_S.index_select(dim=2, index=gS_idx)                                    # [B,H,Gk,Ks]

    # 初始化输出
    inds_K = torch.empty(B, H, Gk, indices_count, dtype=torch.int32, device=device)
    counts_K = torch.full((B, H, Gk), indices_count, dtype=torch.int32, device=device)

    if not add_sink or sink_len <= 0 or indices_count == 0:
        # 直接裁切/循环补齐模板到 indices_count
        if Ks >= indices_count:
            inds_K.copy_(templ[..., :indices_count])
        else:
            rep = math.ceil(indices_count / Ks)
            inds_K.copy_(templ.repeat_interleave(repeats=rep, dim=-1)[..., :indices_count])
        # 边界保护
        inds_K.clamp_(0, max(0, cur_k_len - 1))
        return inds_K, counts_K

    # --- 带 sink 的路径 ---
    # sink 的配额（逐 (B,H,G) 相同），注意上界
    sink_quota = min(int(round(indices_count * sink_quota_frac)), indices_count, sink_len)
    templ_quota = indices_count - sink_quota

    # 1) 写入 sink 前缀：广播写入 [0:sink_quota]
    if sink_quota > 0:
        sink = torch.arange(sink_len, device=device, dtype=torch.int32)
        if sink_len >= sink_quota:
            sink_block = sink[:sink_quota]  # [sink_quota]
        else:
            rep = math.ceil(sink_quota / sink_len)
            sink_block = sink.repeat(rep)[:sink_quota]
        inds_K[..., :sink_quota] = sink_block.view(1, 1, 1, -1)

    if templ_quota == 0:
        inds_K.clamp_(0, max(0, cur_k_len - 1))
        return inds_K, counts_K

    # 2) 从模板中选出不与 sink 冲突的列，数量为 templ_quota：
    #    向量化：把 < sink_len 的位置屏蔽成“大值”，然后用 topk(取最小)一次性选出 templ_quota 个值。
    if sink_len > 0:
        big_val = torch.iinfo(torch.int32).max  # 用 int32 最大值作屏蔽
        templ_masked = torch.where(templ >= sink_len, templ, templ.new_full(templ.shape, big_val))
    else:
        templ_masked = templ

    Ksel = templ_masked.shape[-1]
    if templ_quota <= Ksel:
        # topk 取“最小”的 templ_quota 个值：对负号后取 largest=True
        neg_vals = -templ_masked.to(torch.int64)  # 防止溢出
        topk_pos = torch.topk(neg_vals, k=templ_quota, dim=-1, largest=True).indices  # [B,H,Gk,templ_quota]
        templ_pick = torch.gather(templ_masked, dim=-1, index=topk_pos).to(torch.int32)  # [B,H,Gk,templ_quota]
    else:
        # 先取全部合法项，再循环补齐
        neg_vals = -templ_masked.to(torch.int64)
        topk_pos = torch.topk(neg_vals, k=Ksel, dim=-1, largest=True).indices
        base_pick = torch.gather(templ_masked, dim=-1, index=topk_pos).to(torch.int32)   # [B,H,Gk,Ksel]
        rep = math.ceil(templ_quota / Ksel)
        templ_pick = base_pick.repeat_interleave(repeats=rep, dim=-1)[..., :templ_quota] # [B,H,Gk,templ_quota]

    # 3) 写入模板段
    inds_K[..., sink_quota:sink_quota + templ_quota] = templ_pick

    # 4) 边界保护
    inds_K.clamp_(0, max(0, cur_k_len - 1))
    return inds_K, counts_K


def map_sparse_kv_inds(
    inds_S,      # [B,H,QG,Kkeep] @ scale S
    S, K,
    n_list,      # [n_0,...,n_K]
    cum_S, cum_K,
    hw_list=None,# None => 用 1D比例; 否则用 2D 网格
    sink_first_k: int = 5,    # 按你要求: 一共前5个KV
):
    # A) j, ΔS
    j = torch.bucketize(inds_S, cum_S[1:], right=True)      # each kv index is belong to which scale, e.g. index 0 in scale-0, index 1,2,3,4 in scale-1, ...
    deltaS = inds_S - cum_S[j]      # 每个 kv index 在 scale-S 的各尺度块的偏移量, e.g. index = 7, 在 scale 2 的块内偏移量 = 7 - 5 = 2

    # B) scale-K 的目标块 j'
    jp = (K - (S - j)).clamp_min(0)

    # C) ΔK
    if hw_list is None:
        n   = torch.as_tensor(n_list, device=inds_S.device)
        nj  = n[j]
        njp = n[jp]

        deltaK  = (((deltaS + 0.5) * njp) // nj).clamp_max(njp - 1)
    else:
        h = torch.as_tensor([h for h,_ in hw_list], device=inds_S.device)
        w = torch.as_tensor([w for _,w in hw_list], device=inds_S.device)
        hj,  wj  = h.index_select(0, j),  w.index_select(0, j)
        hjp, wjp = h.index_select(0, jp), w.index_select(0, jp)
        u, v  = deltaS // wj, deltaS % wj
        up    = (u * hjp) // hj
        vp    = (v * wjp) // wj
        deltaK    = (up * wjp + vp).clamp_max(hjp*wjp - 1)

    # D) 还原绝对索引
    inds_K = cum_K[jp] + deltaK

    # E) sink 保护：追加前5个KV索引并截断
    if sink_first_k and sink_first_k > 0:
        # B,H,QG,Kkeep = inds_K.shape
        B,H,QG,Kkeep = inds_S.shape
        sink = torch.arange(min(sink_first_k, int(cum_K[-1].item())), device=inds_K.device, dtype=inds_K.dtype)
        sink = sink.view(1,1,1,-1).expand(B,H,QG,-1)
        inds_K = torch.cat([sink, inds_K], dim=-1)[..., :Kkeep]

    return inds_K.to(torch.int32)  # [B,H,QG,Kkeep]


# ==================
#* index map visual
# ==================
@torch.no_grad()
def selection_mask_from_inds(
    inds_K: torch.Tensor,   # [B, H, Gk, Kk]  每个 query-group 的选 K 索引（已经是跨尺度映射后的）
    q_len: int,             # 目标尺度的 Q 序列长度（这一尺度的 token 数）
    kv_len: int,            # 目标尺度可见的 KV 总长度（历史+当前）
    bm: int,                # query-group 大小（64 或 192）
    clamp_kv_len: Optional[int] = None,  # 额外裁剪上界（一般 = kv_len）
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    输出：mask，形状 [B, H, q_len, kv_len]，元素∈{0,1}，1 表示该 (query_row, key_col) 被选中。
    实现细节：
      - 行到组的映射：row_groups[r] = clamp(floor(r / bm), 0, Gk-1)
      - 向量化展开：inds_per_row = index_select(inds_K, dim=2, index=row_groups)
      - 构建二值图：mask.scatter_(dim=-1, index=inds_per_row, value=1)
    """
    assert inds_K.ndim == 4, f"{inds_K.shape=}"
    B, H, Gk, Kk = inds_K.shape
    device = device or inds_K.device

    # 逐行对应到它属于哪个 query-group
    row_groups = torch.div(torch.arange(q_len, device=device), bm, rounding_mode='trunc')
    row_groups = torch.clamp(row_groups, max=Gk - 1).to(torch.long)          # [q_len]

    # 取出每一行对应组的 K 索引： [B, H, q_len, Kk_sel]
    inds_per_row = inds_K.index_select(dim=2, index=row_groups)              # [B,H,q_len,Kk]
    if clamp_kv_len is None:
        clamp_kv_len = kv_len
    inds_per_row = inds_per_row.clamp_(0, max(0, clamp_kv_len - 1)).long()

    # 构建二值选择图：一次 scatter 完成
    mask = torch.zeros((B, H, q_len, kv_len), dtype=torch.uint8, device=device)
    mask.scatter_(dim=-1, index=inds_per_row, value=1)
    return mask


@torch.no_grad()
def save_attention_selection_visuals(
    inds_K: torch.Tensor,           # [B, H, Gk, Kk]
    q_len: int,
    kv_len: int,
    bm: int,
    out_dir: str,
    prefix: str = "attn_sel",
    downsample_q: int = 1,          # q 维下采样因子（>=1）
    downsample_k: int = 1,          # k 维下采样因子（>=1）
    sink_len: int = 0,              # 若 >0，在图上画一条竖线标注 sink 的右边界
    clamp_kv_len: Optional[int] = None,
):
    """
    为每个 (batch, head) 保存一张 PNG：
      文件名: f"{prefix}_b{b}_h{h}_Q{q_len}_K{kv_len}.png"
    """
    os.makedirs(out_dir, exist_ok=True)
    B, H = inds_K.shape[:2]

    sel = selection_mask_from_inds(
        inds_K=inds_K, q_len=q_len, kv_len=kv_len, bm=bm, clamp_kv_len=clamp_kv_len
    )  # [B,H,q_len,kv_len], uint8

    # 如需下采样，用最近邻（保持二值视觉语义）
    if downsample_q > 1 or downsample_k > 1:
        # 转为 float 后加 channel 维做 interpolate，再阉回到 uint8 显示
        sel_f = sel.float().unsqueeze(2)  # [B,H,1,q,k]
        new_q = math.ceil(q_len / max(1, downsample_q))
        new_k = math.ceil(kv_len / max(1, downsample_k))
        sel_ds = F.interpolate(sel_f, size=(new_q, new_k), mode='nearest').squeeze(2)
        sel = sel_ds.clamp(0, 1).to(torch.uint8)
        q_vis, k_vis = new_q, new_k
        sink_x = int(round(sink_len / max(1, downsample_k))) if sink_len > 0 else -1
    else:
        q_vis, k_vis = q_len, kv_len
        sink_x = sink_len

    # 逐 (B,H) 存文件
    for b in range(B):
        for h in range(H):
            img = sel[b, h].cpu().numpy()  # [q_vis, k_vis], {0,1}

            fig, ax = plt.subplots(figsize=(max(6, k_vis/600*6), max(4, q_vis/600*4)))
            im = ax.imshow(img, interpolation='nearest', aspect='auto')  # 默认灰度 colormap
            ax.set_title(f"b{b}, h{h}  (Q={q_len}, K={kv_len}, bm={bm})")
            ax.set_xlabel("Keys (columns)")
            ax.set_ylabel("Queries (rows)")

            if sink_len > 0:
                ax.axvline(x=sink_x-0.5, linestyle='--')  # 画出 sink 分界

            fig.tight_layout()
            save_path = os.path.join(out_dir, f"{prefix}_b{b}_h{h}_Q{q_len}_K{kv_len}.png")
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
            print(f'{save_path=}')


# ====================
#* local sparse index
# ====================
@torch.no_grad()
def local_window_indices(
    H: int,
    W: int,
    base: int,
    win: Tuple[int, int] = (7, 7),
    dil: Tuple[int, int] = (1, 1),
    padding: str = "clamp",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """
    在单个尺度块 (H x W) 内，为每个 query 生成其二维局部窗口的 KV 绝对列索引。
    返回形状：[H*W, k_local]，其中 k_local = win_h * win_w。
    base 为该块在全局 KV 轴上的起始偏移（即 cum[src]）。
    """
    device = device or torch.device("cpu")
    h_win, w_win = win
    dh, dw = dil

    qy = torch.arange(H, device=device)[:, None].expand(H, W)  # [H,W]
    qx = torch.arange(W, device=device)[None, :].expand(H, W)  # [H,W]

    # 以中心对齐的相对位移
    ry = (h_win - 1) // 2
    rx = (w_win - 1) // 2
    dy = (torch.arange(-ry, h_win - ry, device=device) * dh)  # [h_win]
    dx = (torch.arange(-rx, w_win - rx, device=device) * dw)  # [w_win]
    offy = dy[:, None].expand(h_win, w_win)
    offx = dx[None, :].expand(h_win, w_win)

    ny = (qy[..., None, None] + offy).reshape(H, W, -1)  # [H,W,K_local]
    nx = (qx[..., None, None] + offx).reshape(H, W, -1)  # [H,W,K_local]

    # 边界处理
    if padding == "clamp":
        ny = ny.clamp_(0, H - 1)
        nx = nx.clamp_(0, W - 1)
    elif padding == "circular":
        ny = ny.remainder(H)
        nx = nx.remainder(W)
    elif padding == "reflect":
        def reflect(v, L):
            mod = (v.abs() // (L + 1)) % 2
            v2 = v.abs() % (L + 1)
            return torch.where(mod.bool(), L - v2, v2)
        ny = reflect(ny, H - 1)
        nx = reflect(nx, W - 1)
    else:
        raise ValueError(f"Unknown padding={padding}")

    # 线性化并平移到全局 KV 轴
    lin = (ny * W + nx).view(-1, h_win * w_win)  # [H*W, K_local]
    abs_idx = (lin + int(base)).to(dtype)

    return abs_idx


@torch.no_grad()
def build_crossscale_indices(
    scales, n_list, cum,
    K: int,
    window_spec: Dict[int, Tuple[int, int]],
    padding: str = "clamp",
    device: Optional[torch.device] = None,
    unique: bool = True,
):
    """
    构建「跨尺度」的局部稀疏 KV 绝对列索引（不分组、不压缩），方便直观分析与可视化。

    参数：
      - scales:   各尺度边长 S 的列表，如 [1,2,4,6,8,12,16,20,24,32,40,48,64]
      - K:        目标尺度下标（例如最后尺度 K=12）
      - window_spec: dict，键为 delta（0=self；1=K↔K-1；2=K↔K-2；…），
                     值为窗口大小 (h_win, w_win)，如 {0:(7,7),1:(5,5),2:(3,3)}
      - padding:  "clamp" | "circular" | "reflect"
      - unique:   是否对每个 query 的所有列进行去重（避免 self 与多层 cross 的重复）

    返回：
      - indices_abs: [q_len, k_total]，按行去重并左对齐、尾部 pad 的绝对 KV 列索引（便于画整图）
      - splits:      dict，记录各 delta 在横向拼接前的切片范围（用于对比 self/cross 段）
      - kv_len:      全局 KV 总长度
      - cum:         各尺度块的 KV 累积边界（可用于画竖线）
    """
    device = device or torch.device("cpu")
    # n_list, cum = build_block_boundaries(scales)
    num_scales = len(scales)
    assert 0 <= K < num_scales

    H = W = scales[K]
    q_len = H * W
    kv_len = int(cum[-1].item())

    pieces = []
    splits = {}
    cur = 0
    for delta, win in sorted(window_spec.items()):
        src = K - delta     # scale 11: 11 - 0
        if src < 0:
            continue
        Hs = Ws = scales[src]
        base = cum[src]     # 该 cross/self 块在全局 KV 轴的起点 | e.g. scale 11 4121, scale 12 6425

        # 在 src 块内生成「其自身」的局部邻域
        idx_src = local_window_indices(
            Hs, Ws, int(base.item()), win=win, padding=padding, device=device
        )  # [Hs*Ws, k_local_of_src]

        # 将 K 尺度的每个 query 映射到 src 的最近邻 token，然后取对应行
        if delta == 0:
            if idx_src.shape[0] != q_len:
                raise RuntimeError("Self block q_len mismatch.")
            piece = idx_src  # self：一一对应
        else:
            yK = torch.arange(H, device=device)[:, None].expand(H, W)
            xK = torch.arange(W, device=device)[None, :].expand(H, W)
            yS = (yK * Hs) // H
            xS = (xK * Ws) // W
            linS = (yS * Ws + xS).reshape(-1)  # [q_len]
            piece = idx_src.index_select(0, linS)  # [q_len, k_local_of_src]

        pieces.append(piece)
        splits[delta] = slice(cur, cur + piece.shape[1])
        cur += piece.shape[1]

    indices_abs = (
        torch.cat(pieces, dim=1)
        if pieces
        else torch.empty((q_len, 0), dtype=torch.long, device=device)
    )

    if unique and indices_abs.numel() > 0:
        # 对每行排序 -> unique_consecutive 去重 -> 再 pad 成定长（仅用于可视化/分析）
        sorted_vals, _ = torch.sort(indices_abs, dim=1)
        rows = []
        maxlen = 0
        for r in range(sorted_vals.shape[0]):
            u = torch.unique_consecutive(sorted_vals[r], dim=0)
            rows.append(u)
            if u.numel() > maxlen:
                maxlen = int(u.numel())
        pad_rows = []
        for u in rows:
            if u.numel() < maxlen:
                pad_rows.append(torch.cat([u, u[-1:].repeat(maxlen - u.numel())], dim=0))
            else:
                pad_rows.append(u)
        indices_abs = torch.stack(pad_rows, dim=0)

    return indices_abs, splits, kv_len, cum


def _ceil_to_multiple(x: int, m: int) -> int:
    if m is None or m <= 1:
        return x
    return ((x + m - 1) // m) * m


@torch.no_grad()
def aggregate_group_indices(
    indices_abs: torch.Tensor,
    q_len: int,
    kv_len: int,
    bm: int = 192,
    counts_multiple_of: Optional[int] = None,
    clamp_kv: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """
    将逐 query 的稀疏 KV 绝对列索引 [q_len, k_local_total]
    聚合为按 query-group 的 indices / counts（单批、单头版本）.

    参数:
      - indices_abs: LongTensor [q_len, K_local_total]，每行是该 query 选中的绝对 KV 列索引
      - q_len:       当前尺度的 query 数（例如 64*64=4096）
      - kv_len:      全局 KV 总长度（例如 10521）
      - bm:          query 组大小（Chipmunk/ThunderKittens 常用 192）
      - pad_to:      若给定，则每组的保留列数向上 pad 至该倍数；最终也会将所有组统一 pad 到 `K_pad = max_group_len` 再向上 pad_to
      - counts_multiple_of: 若给定，则每组 counts 先向上取整到该倍数（对某些 kernel 的计数对齐有用）
      - clamp_kv:    是否将所有列索引 clamp 到 [0, kv_len-1]（安全起见默认 True）
      - device:      输出张量设备；若 None 则沿用 indices_abs.device

    返回:
      - group_indices: LongTensor [QG, K_pad]，每组的 KV 列索引（去重、排序、pad 后）
      - group_counts:  IntTensor  [QG]，每组真实的保留列数（若设 counts_multiple_of 则为对齐后的数）
      - group_ranges:  list[(lo, hi))，每组覆盖的 query 行区间（便于调试）

    备注:
      - 这里采用**并集**策略：一个组内所有 query 选到的列索引做 union，保证不漏掉重要列；
      - 若你有“列和分数/重要性分数”，可在 union 后做 Top-K 再 pad；本函数不做 Top-K，仅做去重与对齐。
    """
    device = device or indices_abs.device
    assert indices_abs.dim() == 2 and indices_abs.shape[0] == q_len, \
        f"indices_abs must be [q_len, K], got {indices_abs.shape}"
    if clamp_kv:
        indices_abs = indices_abs.clamp_(0, kv_len - 1)

    # 计算组数与每组的 query 起止
    q_groups = math.ceil(q_len / bm)
    group_ranges: List[Tuple[int,int]] = []
    for g in range(q_groups):
        lo = g * bm
        hi = min((g + 1) * bm, q_len)
        group_ranges.append((lo, hi))       # [(0, 192), (192, 384), ...]

    # 逐组做并集 + 排序 + 对齐
    per_group_lists: List[torch.Tensor] = []
    per_group_counts: List[int] = []

    for (lo, hi) in group_ranges:
        # print(f'\nIn group range [{lo}, {hi})')
        # flatten 该组所有 query 的列索引，然后去重
        flat = indices_abs[lo:hi].reshape(-1)
        uniq = torch.unique(flat)  # 排序 + 去重（升序）
        # print(f'{uniq=}')
        cnt  = int(uniq.numel())
        # print(f'{cnt=}')

        # 对计数做可选对齐（只对 counts 约束；实际 indices 如需 pad，则后面统一 pad）
        cnt_aligned = _ceil_to_multiple(cnt, counts_multiple_of) if counts_multiple_of else cnt
        # print(f'{cnt_aligned=}')

        per_group_lists.append(uniq)
        per_group_counts.append(cnt_aligned)

    # 将所有 group 中的值补齐到 group_count 的数量
    group_count = max(cnt_aligned for cnt_aligned in per_group_counts) if per_group_counts else 0
    starts = torch.stack([t[0] for t in per_group_lists])

    # 最大 index 不能超过 kv_len，若超过，则从左侧补齐
    max_index = torch.tensor(kv_len - 1)
    # 目标每行的最后一个元素：min(start + L - 1, max_val)
    tails = torch.minimum(starts + (group_count - 1), max_index)
    
    # 生成每行的等差序列：[tails - (L-1), ..., tails]
    base = torch.arange(group_count)
    group_indices = (tails[:, None] - (group_count - 1)) + base

    return group_indices, group_count


# Initialized based on sequence shape
singleton_static_mask = None
singleton_video_query_groups = None


class SparseDiffAttn(nn.Module):
    def __init__(
        self, layer_num: int, layer_counter: LayerCounter, 
        use_o_cache: bool = True,
        scales: list = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64],
        spsd_scale: int = 10,
        wind_size: list = [7, 9],    # scale-11 use 5x5 window size and scale-12 use 7x7 window size
        speedup: bool = False,
    ):
        super().__init__()
        self.layer_num = layer_num          # record number of model attn layer 记录这是模型中的第几层Attention
        print(f'\nCurrent init layer is {self.layer_num}')
        self.layer_counter = layer_counter  # todo: diffusion step --> var scale
        self.storage = AttnStorage(layer_num, init_names=['indices', 'out_cache'])              # 该层的专属 cache：indices: 用于存储在稠密步骤中计算出的稀疏模式（即Top-K Key的索引）；out_cache: 用于存储论文公式中的 O_cache，即 O_dense - ΔO_old
        self.mask_shape = [None] * GLOBAL_CONFIG['num_model_invocations_per_inference_step']    # 如果启用了索引压缩（bitpacking），这个变量用来存储原始mask的形状，以便后续解压
        self.use_o_cache = use_o_cache
        if use_o_cache:
            print(f'[Sparse-Attn] Use O_cache is {use_o_cache}, O_t = O_cache + ΔO_new')
        else:
            print(f'[Sparse-Attn] Use O_cache is {use_o_cache}, O_t = ΔO_new')
        self.spsd_scale = spsd_scale
        print(f'    Sparse decision scale is scale-{spsd_scale}')

        attn_config = GLOBAL_CONFIG['attn']
        attn_kernel_config = get_kernel_config_attn()
        bm = attn_kernel_config['bm']
        multiple_of = attn_kernel_config['counts_multiple_of']      # 112

        scales_init = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
        scales = scales_init[:spsd_scale+1]
        print(f'{scales=}')
        # scales = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40]        # for skip last 2 scales

        # wind_size_init = [
        #     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        #     3,      # scale 10 
        #     5,      # scale 11 
        #     7,      # scale 12
        # ]
        wind_size_init = attn_config['win_size']
        wind_size = wind_size_init[len(scales)-len(scales_init):]
        print(f'{wind_size=}')

        self.n_list = [s * s for s in scales_init]
        self.cum = torch.tensor([0] + list(np.cumsum(self.n_list)))     # [0, 1, 5, 21, 57, 121, 265, 521, 921, 1497, 2521, 4121, 6425, 10521]
        # window_spec = {
        #     0: (7, 7),  # self
        #     1: (5, 5),  # K↔K-1
        #     2: (3, 3),  # K↔K-2
        #     3: (3, 3),  # K↔K-3
        # }

        indices_count = int(multiple_of * round((attn_config['top_keys'] * int(self.cum[spsd_scale+1])) / multiple_of))
        print(f'    Sparsity: ratio={attn_config['top_keys']}, {indices_count=}')
        
        assert len(wind_size) == len(scales_init) - (spsd_scale+1), f'w/ local prior sparse, window size list mismatch sparse scales!'
        self.local_inds = {}
        for i in range(len(scales_init) - (spsd_scale+1)):       # spsd_scale (begin 0)
            cur_sps_scale = (spsd_scale+1) + i
            cur_sps_window_spec = {0: (wind_size[i], wind_size[i])}
            cur_sps_cum = self.cum[:cur_sps_scale+2]
            print(f'Scale-{cur_sps_scale} use sparse-attn, local window size is {cur_sps_window_spec}, kv_cum is {cur_sps_cum[-1]}')

            local_indices_abs, _, kv_len, _ = build_crossscale_indices(
                scales=scales_init, n_list=self.n_list, cum=cur_sps_cum,
                K=cur_sps_scale, window_spec=cur_sps_window_spec)
            group_inds, group_count = aggregate_group_indices(
                local_indices_abs,
                q_len=self.n_list[cur_sps_scale], kv_len=kv_len,
                bm=bm, counts_multiple_of=multiple_of)       # scale-11: [q_groups(12), 448], scale-12: [q_groups(22), 672]
            
            print(f'    {group_inds.shape=}, {group_count=}')

            self.local_inds[cur_sps_scale] = (group_inds, group_count)
        
        # Which version dense_colsum_attn should we use?
        # speedup = attn_config['speedup']
        if not speedup:
            print(f'Use ** quality ** dense_colsum_attn')
            self.dense_colsum_attn = dense_colsum_attn_q
        else:
            print(f'Use ** speedup ** dense_colsum_attn')
            self.dense_colsum_attn = dense_colsum_attn_s


    @torch.compiler.disable
    def _fast_attention(
        self, q: Tensor, k: Tensor, v: Tensor, scale,
        inference_step: int, do_full_step: bool,
        scale_ind: int,
        layer_ind: int
    ) -> Tensor:
        """核心调度逻辑

        根据do_full_step标志来决定执行稠密路径还是稀疏路径
        @torch.compiler.disable装饰器告诉编译器不要编译这个Python函数，因为它本身只是一个调度器，真正的性能来自它调用的预编译的chipmunk.ops CUDA/Triton Kernel

        Args:
            q (Tensor): _description_
            k (Tensor): _description_
            v (Tensor): _description_
            inference_step (int): _description_
            do_full_step (bool): _description_

        Returns:
            Tensor: _description_
        """
        attn_config = GLOBAL_CONFIG['attn']
        attn_kernel_config = get_kernel_config_attn()
        bm = attn_kernel_config['bm']
        layer = self.layer_num      # replace by layer_ind
        # print(f'\n****** current {scale_ind=} {layer_ind=} ******')

        multiple_of = attn_kernel_config['counts_multiple_of']
        indices_pad_to = attn_kernel_config['indices_pad_to']
        provider = attn_config['provider']

        # if layer_ind < attn_config['first_n_dense_layers']:
        #     # print(f'first_{layer_ind}_dense_layer')
        #     o, _ = dense_attn(q, k, v, scale)
        #     # --- torch flash-attn ---
        #     # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        #     #     o = F.scaled_dot_product_attention(q, k, v, scale=scale)
        #     return o

        # ─────────── FULL STEP ───────────
        if do_full_step:
            # print(f'FULL STEP: {do_full_step=}')
    
            # if inference_step == 0:
            #     o, lse = dense_attn(q, k, v)    # lse: [bs, num_heads, q_len, 1]
            #     self.storage.set_lse_constants(lse)
            #     return o

            #* ----------- 1. compute colsum ----------
            # o, lse = torch.ops.chipmunk.dense_attn(q, k, v)
            # self.storage.set_lse_constants(lse)
            # print("lse.stride():", tuple(lse.stride()))
            # prev_lse = self.storage.get_lse_constants()
            # print("prev_lse.stride():", tuple(prev_lse.stride()))
            # print(f'dense_colsum_attn Input: {q.shape=} {q.dtype=}, {k.shape=} {k.dtype=}, {v.shape=} {v.dtype=}, {scale=}')
            o, bs = self.dense_colsum_attn(q, k, v, scale)
            # --- naive version ---
            # o, bs = naive_dense_colsum_attn(q, k, v, scale)
            # o, bs = naive_dense_colsum_attn2(q, k, v, scale)
            # print(f'dense_colsum_attn: {o.shape=}, {bs.shape=}')

            indices_count = int(multiple_of * round((attn_config['top_keys'] * k.shape[-2]) / multiple_of))
            # print(f'****** {indices_count=} ******')
            
            #* ---------- 2. TopK and get sparse KV index ----------
            inds = torch.topk(bs, k=indices_count, dim=-1).indices; #print(f'TopK: {inds.shape}')      # e.g. [batch, num_heads, q_blocks(9), 640]

            # --- Visualize decision scale index ---
            save_attention_selection_visuals(
                inds_K=inds, q_len=q.shape[-2], kv_len=k.shape[-2], bm=bm,
                out_dir=f'work_dir/analysis/index_visual/scale_10/layer-{layer_ind}'
            )

            counts = torch.full(
                (q.shape[0], q.shape[1], triton.cdiv(q.shape[-2], bm)), 
                indices_count, device=q.device, dtype=torch.int32
            )
            # Pad the stride, but not the shape, of indices so that the TMA stride gets aligned to 16 bytes
                # padding_amount: 另一个关键的硬件优化。这里对索引进行填充，是为了确保内存访问满足NVIDIA Hopper架构中 TMA（张量内存加速器）的对齐要求（通常是128位/16字节），从而最大化数据传输效率
            # --cross-attn--> 计算 padding_amount 时改用 N_k
            padding_amount = (k.shape[-2] - indices_count + indices_pad_to - 1) // indices_pad_to * indices_pad_to
            inds = torch.cat(
                [inds, torch.empty((*counts.shape, padding_amount), device=q.device, dtype=torch.int32)], 
                dim=-1
            ).to(torch.int32)       # scale 10: torch.Size([2, 16, 9, 4124]) but k_len = 4121
            inds = inds[:,:,:,:k.shape[-2]]
            
            self.storage.set_indices(inds)
            self.storage.set_counts(counts)

            if self.use_o_cache:
                if provider == 'cuda': o_cache = o.clone()
                else: o_cache = o
                
                #* ---------- 3. sparse-attn and cache O_cache ----------
                    # O_cache = O_dense + (-1)ΔO
                o_cache = csp_attn(q, k, v, scale, inds, counts, o_cache, -1)

                self.storage.set_out_cache(o_cache)
            
            return o

        # ─────────── SPARSE STEP ───────────
        # print(f'SPARSE STEP: {do_full_step=}')
        # ------ 1. load sparse patterns ------
        inds   = self.storage.get_indices()     # after scale10, [2, num_heads, 9, 4142]
        counts = self.storage.get_counts()      # after scale10, [2, num_heads, 9], value=640
        s_indices_count = int(counts[0][0][0])

        if inds.shape[0] != q.shape[0]:
            inds = inds[:q.shape[0]]; counts = counts[:q.shape[0]]      # for skip last 2 scales

        # *cross-scale map
        per_scale_tokens = [1, 4, 16, 36, 64, 144, 256, 400, 576, 1024, 1600, 2304, 4096]
        # cur_scale = per_scale_tokens.index(q.shape[-2])
        # --- q_groups dim nearest map ---
        inds, counts = map_inds_counts_across_scales(
            inds, counts, cur_q_len=q.shape[-2], cur_k_len=k.shape[-2],
            bm=bm, indices_pad_to=indices_pad_to)
            # --> simple version, only map q_group dim
        # Gk = (q.shape[-2] + bm - 1) // bm
        # Gs = inds.shape[-2]
        # gk_centers = torch.arange(Gk, device=inds.device, dtype=torch.float32) + 0.5
        # gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).to(torch.int32).clamp_(0, Gs - 1)
        # inds = inds.index_select(dim=2, index=gS_idx)       # scale 11: [batch, num_heads, 12, 4121] only [batch, num_heads, 12, :672] useful

        # --- for visual sparse index ---
        # save_attention_selection_visuals(
        #     inds_K=inds, q_len=q.shape[-2], kv_len=k.shape[-2], bm=bm,
        #     out_dir=f'work_dir/analysis/index_visual/q_map_scale_{int(math.sqrt(q.shape[-2]))}/layer-{layer_ind}'
        # )

        #* --- v2 kv dim map --- Correct but not eval
        # cum = np.cumsum([0] + per_scale_tokens).tolist() # [0, 1, 5, 21, 57, 121, 265, 521, 921, 1497, 2521, 4121, 6425, 10521]
        # cum_tensor = torch.tensor(cum).to(inds.device)
        # inds = map_sparse_kv_inds(
        #     inds_S=inds,
        #     S=self.spsd_scale, K=scale_ind,
        #     n_list=self.n_list,
        #     cum_S=self.cum.to(inds.device), cum_K=self.cum.to(inds.device),
        #     sink_first_k=sum(self.n_list[:scale_ind-self.spsd_scale])
        # )
        # indices_count = s_indices_count

        #! --- add current scale loacal prior ---
        local_inds, local_count = self.local_inds[scale_ind]
        local_inds = local_inds.to(device=inds.device, dtype=inds.dtype)
        inds = torch.cat(
            [inds, local_inds.expand(q.shape[0], q.shape[1], -1, -1)], dim=-1)

        indices_count = s_indices_count + local_count        # add local sparse index
        assert indices_count % multiple_of == 0, f"after add local sparse index, mismatch {multiple_of=}"

        # --- Visualize decision scale index ---
        if scale_ind == 10:
            save_attention_selection_visuals(
                inds_K=inds, q_len=q.shape[-2], kv_len=k.shape[-2], bm=bm,
                out_dir=f'work_dir/analysis/index_visual/kv_map_scale_{int(math.sqrt(q.shape[-2]))}-{attn_config['top_keys']}/layer-{layer_ind}'
            )

        # inds change, need pad again
        counts = torch.full(
            (q.shape[0], q.shape[1], triton.cdiv(q.shape[-2], bm)), 
            indices_count, device=q.device, dtype=torch.int32
        )
        padding_amount = (k.shape[-2] - indices_count + indices_pad_to - 1) // indices_pad_to * indices_pad_to
        inds = torch.cat(
            [inds, torch.empty((*counts.shape, padding_amount), device=q.device, dtype=torch.int32)], 
            dim=-1
        ).to(torch.int32)
        inds = inds[:,:,:,:k.shape[-2]]

        #! --- add current scale loacal prior ---
        if scale_ind != 12:
            self.storage.set_indices(inds)      # cache scale-11 indices for scale-12
            self.storage.set_counts(counts)
        
        # ------ 2. load O_cache ------
        if self.use_o_cache:
            o = self.storage.get_out_cache()
            if provider == 'cuda': o = o.clone()
            
            if o.shape[0] != q.shape[0]: o = o[:q.shape[0]]     # for skip last 2 scales
            # --- upscale o_cache ---
            o = upsample_Ocache(o, o.shape[-2], q.shape[-2], upsample_mode='bilinear').to(v.dtype)
        else:
            o = torch.zeros_like(q)

        # ------ 3. sparse-attn Δ and update o ------
            # O_t = O_cache + (1)ΔO_new
        # print(f'%%% before csp_attn %%% {q.shape=}, {k.shape=}, {scale=}, {inds.shape=}')
        o = csp_attn(q, k, v, scale, inds, counts, o, 1)
        
        return o
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, scale, do_full_step: bool = False, inference_step: int = 0, 
                scale_ind: int = 0, layer_ind: int = 0) -> Tensor:
        # check if Chipmunk is enabled
        if not GLOBAL_CONFIG['attn']['is_enabled']:
            out = F.scaled_dot_product_attention(q, k, v)
            self.layer_counter.increment()
            return out

        # 决定当前是稠密还是稀疏步骤
        # do_full_step = self.layer_counter.should_do_full_attn_step()
        inference_step = self.layer_counter.cur_inference_step

        # 调用核心逻辑
        out = self._fast_attention(q, k, v, scale, inference_step, do_full_step, 
                                   scale_ind, layer_ind)

        # 更新全局计数器
        self.layer_counter.increment()
        self.storage.complete_cur_layer()
        # print(f'### update layer_counter, {self.layer_counter.cur_inference_step=} ###')
        
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    # torch.set_default_device('cuda')
    # torch.set_default_dtype(torch.bfloat16)

    # for seqlen in range(4480, 4480+112*10, 112):
    #     for is_contiguous in [False, True]:
    #         b, h, n, d = 1, 24, seqlen, 128
    #         n_groups = (n + 192 - 1) // 192

    #         def make_tensor(shape, is_contiguous, fill_value=None):
    #             if is_contiguous:
    #                 new_vec = torch.randn(shape)
    #             else:
    #                 new_shape = (shape[2], shape[0], shape[1], shape[3])
    #                 new_vec = torch.randn(*new_shape)
    #                 new_vec = new_vec.permute(1, 2, 0, 3)
    #             if fill_value is not None:
    #                 new_vec.fill_(fill_value)
    #             return new_vec
                    

    #         q = make_tensor((b, h, n, d), is_contiguous)
    #         k = make_tensor((b, h, n, d), is_contiguous)
    #         v = make_tensor((b, h, n, d), is_contiguous)
    #         o = make_tensor((b, h, n, d), True, fill_value=0)

    #         indices = torch.arange(n, dtype=torch.int32).repeat((b, h, n_groups, 1)).contiguous()
    #         counts = torch.full((b, h, n_groups), n, dtype=torch.int32)

    #         torch.ops.chipmunk.csp_attn(q, k, v, o, indices, counts, 1)

    #         o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    #         o_max_diff = (o - o_ref).abs().max()
    #         o_mean_diff = (o - o_ref).abs().mean()

            
    #         print(f"is_contig={is_contiguous}, seqlen: {seqlen} (% 192 = {seqlen % 192}, % 112 = {seqlen % 112}), o_max_diff: {o_max_diff:.2f}, o_mean_diff: {o_mean_diff:.2f}")

    # --- sparse index mapping test ---
    # import torch
    # import numpy as np
    # from pprint import pprint

    # scales = [1,2,4,6,8,12,16,20,24,32,40,48,64]
    # n_list = [s*s for s in scales]       # per-scale token number: [1, 4, 16, 36, 64, 144, 256, 400, 576, 1024, 1600, 2304, 4096]
    # # n_list = torch.tensor(n_list)
    # print(f'{n_list=}')

    # cum = np.cumsum([0]+n_list).tolist() # [0, 1, 5, 21, 57, 121, 265, 521, 921, 1497, 2521, 4121, 6425, 10521]
    # cum_tensor = torch.tensor(cum, dtype=torch.long)

    # S = 10
    # B, H, QG, Kkeep = 1, 1, 2, 8
    # torch.manual_seed(0)

    # inds_S = torch.randint(0, cum[S+1], (B, H, QG, Kkeep))
    # inds_S[0][0][0] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    # print(f'S={S} indices sample:')
    # pprint(inds_S.tolist())

    # K = 11
    # sink_first_k = sum(n_list[:K-S])
    # print(f'{sink_first_k=}')

    # inds_K_11 = map_sparse_kv_inds(
    #     inds_S=inds_S,
    #     S=S, K=K,
    #     n_list=n_list,
    #     cum_S=cum_tensor, cum_K=cum_tensor,
    #     sink_first_k=sink_first_k,   # 前5个KV为sink保护
    # )

    # --- self-attn local sparse index ---
    scales = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
    n_list = [s * s for s in scales]
    cum = torch.tensor([0] + list(np.cumsum(n_list)))     # [0, 1, 5, 21, 57, 121, 265, 521, 921, 1497, 2521, 4121, 6425, 10521]

    K = 12
    window_spec = {
        0: (7, 7),  # self
    }

    indices_abs_11, splits, kv_len, cum = build_crossscale_indices(
        scales=scales, n_list=n_list, cum=cum,
        K=K, window_spec=window_spec,
        padding="clamp",
        # device=torch.device("cuda"),
        unique=True,  # 行级去重，便于可视化
    )

    H = W = scales[K]
    q_len = H * W
    print("indices_abs:", indices_abs_11.shape)
    print("splits:", {k: (v.start, v.stop) for k, v in splits.items()})
    print("kv_len:", kv_len)

    bm = 192
    multiple_of = 112
    indices_pad_to = 4

    group_inds, group_counts = aggregate_group_indices(
        indices_abs_11, q_len, kv_len, bm=192, counts_multiple_of=multiple_of)

    print("group_inds:",  group_inds.shape)   # [QG, K_pad]
    print("group_counts:", group_counts.shape, "max/min:", int(group_counts.max()), int(group_counts.min()))

    



    # # 可视化整图（不降采样）
    # mask = visualize_full_mask(
    #     indices_abs,
    #     q_len=q_len,
    #     kv_len=kv_len,
    #     cum=cum,
    #     title=f"Cross-scale local neighborhoods @ K={K} (self + K-1..K-3)",
    # )

    # # 也可以挑几行看
    # rows = [(H // 2) * W + (W // 2), 0, q_len - 1, np.random.randint(0, q_len)]
    # visualize_rows(mask, cum, rows)

    # # 打印其中一行（中心 query）的部分绝对 KV 列索引以便核对
    # center_q = (H // 2) * W + (W // 2)
    # sel = torch.unique(indices_abs[center_q]).tolist()
    # print(f"Center query ({center_q}) selected {len(sel)} unique KV columns. First 20:", sel[:20])





