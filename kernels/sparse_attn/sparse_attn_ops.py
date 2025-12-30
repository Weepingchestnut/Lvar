import math
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

import kernels.sparse_attn.triton as sattn_triton
from .sparse_attn_config import GLOBAL_CONFIG, get_kernel_config_attn
# from flag_attn import flash_attention


def pad_qkvo_tensor(tensor, pad_to):
    n = tensor.shape[-2]
    padded_n = ((n + pad_to - 1) // pad_to) * pad_to
    # IMPORTANT: Do not use torch.empty here, it will cause NaN in the Triton kernel!
    padded_tensor = torch.zeros(tensor.shape[:-2] + (padded_n, tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[..., :n, :] = tensor
    return padded_tensor


def dense_attn(q, k, v, scale=None):
    """support cross-attention"""
    
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        pad_to = get_kernel_config_attn()['bm']
        # ------ no padding ------ efficient
        o, lse = sattn_triton.dense_attn(
            q, k, v,
            scale=scale
        )
        # ------ padding ------
        # o, lse = sattn_triton.dense_attn(
        #     q, 
        #     # k, v,
        #     pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to),
        #     kv_true_len=k.shape[-2],
        #     scale=scale
        # )
        # ------ flag-attn version ------
        # o = flash_attention(
        #     q, 
        #     # pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to),     # !padding will make img inconsistent
        #     k, v,
        #     causal=False, sm_scale=scale
        # )
        
        assert type(lse) == tuple, "LSE must be a tuple"
        assert lse[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
        assert lse[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
        # if q.shape[-2] >= 1600:
        #     print(f'M: head-0 query-0 {lse[0][0][0][0]}, head-0 query-1 {lse[0][0][0][1]}, head-0 query-10 {lse[0][0][0][10]}, head-0 query-100 {lse[0][0][0][100]}')
        #     print(f'L: head-0 query-0 {lse[1][0][0][0]}, head-0 query-1 {lse[1][0][0][1]}, head-0 query-10 {lse[1][0][0][10]}, head-0 query-100 {lse[1][0][0][100]}')
    else:
        o, lse = torch.ops.chipmunk.dense_attn(q, k, v)
        assert lse.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
    
    assert o.shape == q.shape, "Output shape mismatch"
    
    return o, lse


# ------------
# for speedup
# ------------
def dense_colsum_attn_s(q, k, v, scale=None):
    """
    Compute variable length attention in ThunderKittens.
    """

    provider = GLOBAL_CONFIG['attn']['provider']
    pad_to = get_kernel_config_attn()['bm']

    if provider == 'cuda':
        # CUDA implementation
        # assert p.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "P shape mismatch - p: {}, q: {}".format(p.shape, q.shape)
        # --- 2 kernel 2 pass get colsum ---
        o, lse = torch.ops.chipmunk.dense_attn(q, k, v)     # lse [bs, heads, q_len, 1]
        # print("lse.stride():", tuple(lse.stride()))
        # p = lse.squeeze(-1).contiguous().unsqueeze(-1)
        # print("p.stride():", tuple(p.stride()))

        # if p.stride(1) % 4 != 0:
        #     B, Hq, Q, one = p.shape
        #     Q_pad = ((Q + 3) // 4) * 4
        #     p_aligned = torch.empty_strided(
        #         (B, Hq, Q, 1), (Hq*Q_pad, Q_pad, 1, 1),
        #         dtype=p.dtype, device=p.device
        #     )
        #     # 只拷贝有效区
        #     p_aligned[:, :, :Q, :] = p
        #     p = p_aligned

        _, cs, l = torch.ops.chipmunk.dense_colsum_attn(q, k, v, lse)
        assert l.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)

    else:
        # Triton implementation
        if q.shape[-2] % pad_to == 0 and k.shape[-2] % pad_to == 0 and v.shape[-2] % pad_to == 0:
            o, cs, l = sattn_triton.dense_colsum_attn(q, k, v)
        else:
            # o, cs, l = sattn_triton.dense_colsum_attn(q, pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to))
            # cs = cs[..., :k.shape[2]]       # todo: is it True? | cs [2, num_heads, q_blocks, kv_len]
            # --- 1 kernel 2 pass get colsum ---
            o, cs, l = sattn_triton.dense_colsum_attn(q, k, v, scale=scale)

        assert l[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)
        assert l[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[1].shape, q.shape)

    assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
    assert cs.shape == (q.shape[0], q.shape[1], (q.shape[-2] + pad_to - 1) // pad_to, k.shape[2]), "CS shape mismatch - cs: {}, q: {}".format(cs.shape, q.shape)
    
    return o, cs
    # return o, cs, l


# -----------------------
# for Generation quality
# -----------------------
def dense_colsum_attn_q(q, k, v, scale=None):
    """
    Compute variable length attention in ThunderKittens.
    """

    provider = GLOBAL_CONFIG['attn']['provider']
    pad_to = get_kernel_config_attn()['bm']

    if provider == 'cuda':
        # CUDA implementation
        # assert p.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "P shape mismatch - p: {}, q: {}".format(p.shape, q.shape)
        # --- 2 kernel 2 pass get colsum ---
        o, _ = torch.ops.chipmunk.dense_attn(q, k, v)
        _, cs, _ = sattn_triton.dense_colsum_attn(q, k, v, scale=scale)     #? Using Triton correctly cs
    else:
        # Triton implementation
        if q.shape[-2] % pad_to == 0 and k.shape[-2] % pad_to == 0 and v.shape[-2] % pad_to == 0:
            o, cs, l = sattn_triton.dense_colsum_attn(q, k, v)
        else:
            # o, cs, l = sattn_triton.dense_colsum_attn(q, pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to))
            # cs = cs[..., :k.shape[2]]       # todo: is it True? | cs [2, num_heads, q_blocks, kv_len]
            # --- 1 kernel 2 pass get colsum ---
            o, cs, l = sattn_triton.dense_colsum_attn(q, k, v, scale=scale)

        assert l[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)
        assert l[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[1].shape, q.shape)

    assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
    assert cs.shape == (q.shape[0], q.shape[1], (q.shape[-2] + pad_to - 1) // pad_to, k.shape[2]), "CS shape mismatch - cs: {}, q: {}".format(cs.shape, q.shape)
    
    return o, cs


# ------ only Triton ------
# def dense_colsum_attn(q, k, v, scale=None):
#     """
#     Compute variable length attention in ThunderKittens.
#     """

#     provider = GLOBAL_CONFIG['attn']['provider']
#     pad_to = get_kernel_config_attn()['bm']

#     # Triton implementation
#     if q.shape[-2] % pad_to == 0 and k.shape[-2] % pad_to == 0 and v.shape[-2] % pad_to == 0:
#         o, cs, l = sattn_triton.dense_colsum_attn(q, k, v, scale=scale)
#     else:
#         # o, cs, l = sattn_triton.dense_colsum_attn(q, pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to))
#         # cs = cs[..., :k.shape[2]]       # todo: is it True? | cs [2, num_heads, q_blocks, kv_len]
#         # --- 1 kernel 2 pass get colsum ---
#         o, cs, l = sattn_triton.dense_colsum_attn(q, k, v, scale=scale)

#     assert l[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)
#     assert l[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[1].shape, q.shape)

#     assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
#     assert cs.shape == (q.shape[0], q.shape[1], (q.shape[-2] + pad_to - 1) // pad_to, k.shape[2]), "CS shape mismatch - cs: {}, q: {}".format(cs.shape, q.shape)
    
#     return o, cs, l


@torch.no_grad()
def naive_dense_colsum_attn(q, k, v, scale=None, q_group_size=192):
    B, H, q_len, D = q.shape
    k_len = k.size(-2)

    scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale

    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    o = attn_weight @ v

    q_groups = (q_len + q_group_size - 1) // q_group_size
    if q_groups * q_group_size != q_len:
        q_pad_len = q_groups * q_group_size - q_len
        attn_pad = F.pad(attn_weight, (0, 0, 0, q_pad_len))
    else:
        attn_pad = attn_weight
    
    attn_g = attn_pad.view(B, H, q_groups, q_group_size, k_len)
    cs = attn_g.sum(dim=3).to(v.dtype)
    
    return o, cs


@torch.no_grad()
def naive_dense_colsum_attn2(
    q: torch.Tensor,                 # [B, qo_heads, q_len, d]
    k: torch.Tensor,                 # [B, kv_heads, kv_len, d]
    v: torch.Tensor,                 # [B, kv_heads, kv_len, d]
    scale: Union[int, float, None] = None,
    q_group_size: int = 192,           # 与 kernel 的 q_g 计算保持一致（FUSE_REDUCE=True）
    out_dtype: Optional[torch.dtype] = None,    # 若为 None，跟随 v.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    朴素参考实现：
      返回 (o, cs, l_vec)
        o:     [B, qo_heads, q_len, d]，标准密集注意力输出
        cs:    [B, qo_heads, q_groups, kv_len]，按查询组聚合后的列和矩阵
        l_vec: [B, qo_heads, q_len, 1]，每行 softmax 分母的倒数（自然底 e）
    """
    assert q.ndim == k.ndim == v.ndim == 4, "Q/K/V must be [B, H, N, D]"
    B, qo_h, q_len, d = q.shape
    Bk, kv_h, kv_len, dk = k.shape
    Bv, kv_h2, kv_len2, dv = v.shape
    assert B == Bk == Bv, "batch mismatch"
    assert d == dk == dv, "head_dim mismatch"
    assert kv_h == kv_h2, "KV heads mismatch"
    assert q.dtype == k.dtype == v.dtype or True, "dtypes can differ but beware precision"
    assert qo_h % kv_h == 0, "qo_heads must be a multiple of kv_heads"
    hr = qo_h // kv_h

    # 将 K/V 在“组内”复制到 QO heads 维度（广播到 qo_heads）
    # 形状变换: [B, kv_h, kv_len, d] → [B, kv_h, hr, kv_len, d] → [B, qo_h, kv_len, d]
    k_exp = k.unsqueeze(2).expand(B, kv_h, hr, kv_len, d).reshape(B, qo_h, kv_len, d)
    v_exp = v.unsqueeze(2).expand(B, kv_h, hr, kv_len, d).reshape(B, qo_h, kv_len, d)

    # 标准缩放点积注意力（使用 e 底数的稳定 softmax）
    # scores: [B, qo_h, q_len, kv_len]
    sm_scale = 1/math.sqrt(d) if scale is None else scale
    scores = torch.matmul(q.to(torch.float32), k_exp.transpose(-1, -2).to(torch.float32)) / sm_scale

    # log-sum-exp：m, L
    # m = scores.max(dim=-1, keepdim=True).values                          # [B, qo_h, q_len, 1]
    # exps = torch.exp(scores - m)                                         # [B, qo_h, q_len, kv_len]
    # L = exps.sum(dim=-1, keepdim=True)                                   # [B, qo_h, q_len, 1]
    # attn = exps / L                                                      # [B, qo_h, q_len, kv_len]
    attn = torch.softmax(scores, dim=-1)

    # 输出 o
    o = torch.matmul(attn, v_exp.to(torch.float32))                      # [B, qo_h, q_len, d]
    if out_dtype is None:
        out_dtype = v.dtype
    o = o.to(out_dtype)

    # 列和矩阵 cs：按查询组把 attn 沿 q_len 聚合到 q_groups
    q_groups = (q_len + q_group_size - 1) // q_group_size
    if q_groups * q_group_size != q_len:
        # 为了实现方便，可做 0-padding 再重塑；不影响求和（因为 padding 行是 0）
        pad_len = q_groups * q_group_size - q_len
        attn_pad = torch.nn.functional.pad(attn, (0, 0, 0, pad_len))     # pad queries 维
    else:
        attn_pad = attn
    # 现在把 queries 维重塑为 [q_groups, group_size]，再在 group_size 维上求和
    attn_g = attn_pad.view(B, qo_h, q_groups, q_group_size, kv_len)
    cs = attn_g.sum(dim=3)                                               # [B, qo_h, q_groups, kv_len]
    cs = cs.to(out_dtype)  # 与 kernel 保持 bf16/半精输出习惯

    # l_vec：每行 softmax 分母的倒数（自然底 e）
    # Kernel 用的是 base-2 路径，但数值上与 1/sum(exp(scores)) 等价
    # l_vec = (1.0 / (torch.exp(m) * L)).to(torch.float32)                 # [B, qo_h, q_len, 1]
    # return o, cs, l_vec
    return o, cs


def csp_attn(q, k, v, scale, indices, indices_counts, o, o_scale):
    # Ignore the n_groups dimension in Python - the kernel will also double check for us!
    assert indices.shape == (q.shape[0], q.shape[1], indices.shape[2], k.shape[-2]), "Indices shape mismatch - indices: {}, q: {}, k: {}".format(indices.shape, q.shape, k.shape)
    assert indices_counts.shape == indices.shape[:-1], "Indices counts shape mismatch - indices_counts: {}, indices: {}".format(indices_counts.shape, indices.shape)
    assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
    
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        pad_to = get_kernel_config_attn()['bm']
        o_delta, _ = sattn_triton.csp_attn(
            q, k, v,
            # pad_qkvo_tensor(k, pad_to), 
            # pad_qkvo_tensor(v, pad_to), 
            indices, indices_counts,
            scale=scale,
        )
        o = o + o_delta * o_scale
    else:
        torch.ops.chipmunk.csp_attn(q, k, v, o, indices, indices_counts, o_scale)
    
    return o

# __all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']


# ==========
# bitpack.py
# ==========
@torch.compile(dynamic=False)
def bitpack(mask: torch.Tensor):
    r"""
    Compresses a boolean tensor into a bit-packed uint8 tensor in parallel on the GPU.
    Each output byte encodes 8 bits (True or False) from the input tensor, in little-endian order.

    Args:
        mask (torch.Tensor): A boolean tensor to compress. Must be on the GPU.

    Returns:
        (torch.Tensor, Tuple[int, ...]):
            A tuple of:
            - A 1-D torch.uint8 tensor of length ceil(numel(mask) / 8)
              storing the packed bits on the GPU.
            - The original shape of the mask tensor (for later unpacking).
    """
    original_shape = mask.shape
    # Flatten the tensor
    flat_mask = mask.flatten()
    n = flat_mask.numel()

    # Number of bits we need to pad so that we can reshape into 8 columns
    pad_size = (-n) % 8  # same as: (8 - (n % 8)) % 8

    # Zero-pad if necessary
    flat_mask = torch.cat([flat_mask, flat_mask.new_zeros(pad_size)])

    # Reshape to [N/8, 8], cast to uint8
    flat_mask = flat_mask.view(-1, 8).to(torch.uint8)

    # For each column j, we multiply by 2^j and sum across columns
    # shifts = [1, 2, 4, 8, 16, 32, 64, 128]
    shifts = (2 ** torch.arange(8, dtype=torch.uint8, device=flat_mask.device)).view(1, -1)
    packed = (flat_mask * shifts).sum(dim=1, dtype=torch.uint8).contiguous()  # [N/8]

    return packed, original_shape


@torch.compile(dynamic=False)
def bitunpack(packed: torch.Tensor, original_shape: Tuple[int, ...]):
    r"""
    Decompresses a bit-packed tensor (uint8) back to a boolean tensor in parallel on the GPU.

    Args:
        packed (torch.Tensor): A 1-D bit-packed tensor of type torch.uint8 on the GPU.
        original_shape (Tuple[int, ...]): The original shape of the boolean tensor.

    Returns:
        torch.Tensor: A boolean tensor of shape original_shape.
    """
    # Compute total number of bits needed
    total_bits = 1
    for dim in original_shape:
        total_bits *= dim

    # Expand the packed bytes to 8 bits each
    # shifts = [1, 2, 4, 8, 16, 32, 64, 128]
    shifts = (2 ** torch.arange(8, dtype=torch.uint8, device=packed.device)).view(1, -1)
    
    # (packed.unsqueeze(1) >> shift) & 1 gives bits; shape => [N_bytes, 8]
    bits_2d = ((packed.unsqueeze(1) & shifts) > 0).to(torch.bool)

    # Flatten and truncate if there was padding
    bits = bits_2d.view(-1)[:total_bits]

    # Reshape to the original shape
    return bits.view(*original_shape)


# =============
# indexed_io.py
# =============
def copy_indices(
    bm_fc1: torch.Tensor, 
    bm_mid_cache: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor
) -> None:
    torch.ops.chipmunk.copy_indices(bm_fc1, bm_mid_cache, indices, counts)


def topk_indices(
    activations: torch.Tensor, 
    indices_out: torch.Tensor, 
    counts_out: torch.Tensor, 
    sparsity_amount: float, 
    multiple_of: int, 
    rk: float
) -> None:
    torch.ops.chipmunk.topk_indices(activations, indices_out, counts_out, sparsity_amount, multiple_of, rk)


def scatter_add(
    packed: torch.Tensor, 
    unpacked: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor, 
    num_sms: int
) -> None:
    torch.ops.chipmunk.csp_scatter_add(packed.unsqueeze(0), unpacked.unsqueeze(0), indices.unsqueeze(0), counts.unsqueeze(0), num_sms)


def mask_to_indices(
    mask: torch.Tensor,
    multiple_of: int,
    pad_to_multiple_of: int
) -> List[torch.Tensor]:
    return torch.ops.chipmunk.mask_to_indices(mask, multiple_of, pad_to_multiple_of)


# if __name__ == "__main__":
#     test_dense_colsum_attn()
