# import os
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
from torch.nn import functional as F
import math
import triton
import triton.language as tl

from kernels.sparse_attn.sparse_attn_config import get_kernel_config_attn

DEVICE = 'cuda'

cdiv = lambda a, b: (a + b - 1) // b

# BLOCK_M is pinned to 64: num_qg_per_indices_group = bm // 64 maps three 64-row
# programs onto one bm=192 sparse-indices group. Only stages / warps are tuned.
configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=s, num_warps=w) \
    for s in [2, 3, 4]\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.jit
def _sparse_attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_base, V_base,  #
                    stride_k_seqlen, stride_kk, stride_v_seqlen, stride_vn,  #
                    spi_row, stride_spi_col, sparsity_count,  #
                    qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_n: tl.constexpr, offs_d: tl.constexpr,  #
                    N_CTX_KV: tl.constexpr, MASK_KV: tl.constexpr, fp8_v: tl.constexpr):
    # loop-split over the selected (gathered) keys: STAGE 3 sweeps the full
    # BLOCK_N tiles of sparsity_count (no tail mask); STAGE 4 the partial tail.
    if STAGE == 3:
        lo, hi = 0, (sparsity_count // BLOCK_N) * BLOCK_N
    else:
        lo, hi = (sparsity_count // BLOCK_N) * BLOCK_N, sparsity_count
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        spi = spi_row + (start_n + offs_n) * stride_spi_col
        if MASK_KV:
            valid_sparse = start_n + offs_n < sparsity_count
            sparsity_indices = tl.load(spi, mask=valid_sparse, other=0)
        else:
            sparsity_indices = tl.load(spi)
        tl.device_assert(tl.max(sparsity_indices) < N_CTX_KV)
        tl.device_assert(tl.min(sparsity_indices) >= 0)

        # -- gather K, compute qk --
        k_ptrs = K_base + sparsity_indices[None, :] * stride_k_seqlen + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        # Sparse columns are KV positions; mask the partial sparse-count tail before
        # the row max so padded gather slot 0 cannot affect softmax statistics.
        if MASK_KV:
            qk = tl.where(valid_sparse[None, :], qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i --
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        v_ptrs = V_base + sparsity_indices[:, None] * stride_v_seqlen + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    return acc, l_i, m_i


# NOTE: N_CTX_KV dropped from the autotune key — it tracks per-video k_len and
# re-autotuning every video leaks the sparse-scale activations via retained
# OutOfResources tracebacks. Reuse the config tuned on fixed N_CTX_Q/HEAD_DIM.
@triton.autotune(list(filter(keep, configs)), key=["N_CTX_Q", "HEAD_DIM"])
@triton.jit
def _sparse_attn_fwd(Q, K, V, sm_scale, M, L, Out, Out_accum, Out_scale: tl.constexpr,  #
              sparsity_indices, sparsity_counts,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_spiz, stride_spih, stride_spim, stride_spin,  #
              stride_spcz, stride_spch, stride_spcm,  #
              Z, H_Q, H_KV, N_CTX_Q, N_CTX_KV,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              num_qg_per_indices_group: tl.constexpr,  #
              fp8_v: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h = off_hz % H_Q
    # Sparse indices are per Q head/group but gather KV rows from the mapped KV head.
    off_h_kv = off_h // (H_Q // H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h_kv.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h_kv.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    spi_offset = off_z.to(tl.int64) * stride_spiz + off_h.to(tl.int64) * stride_spih
    spc_offset = off_z.to(tl.int64) * stride_spcz + off_h.to(tl.int64) * stride_spch

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    indices_group = start_m // num_qg_per_indices_group

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qo_mask = (offs_m < N_CTX_Q)[:, None]
    q = tl.load(q_ptrs, mask=qo_mask, other=0.0)

    # per-group sparse count + the row of selected indices for this group
    sparsity_count = tl.load(sparsity_counts + spc_offset + indices_group * stride_spcm)
    spi_row = sparsity_indices + spi_offset + indices_group * stride_spim

    K_base = K + k_offset
    V_base = V + v_offset

    # full gather tiles (unmasked) then the partial tail (masked)
    acc, l_i, m_i = _sparse_attn_fwd_inner(acc, l_i, m_i, q,  #
                                    K_base, V_base,  #
                                    stride_kn, stride_kk, stride_vk, stride_vn,  #
                                    spi_row, stride_spin, sparsity_count,  #
                                    qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    3, offs_n, offs_d, N_CTX_KV, False, fp8_v)
    acc, l_i, m_i = _sparse_attn_fwd_inner(acc, l_i, m_i, q,  #
                                    K_base, V_base,  #
                                    stride_kn, stride_kk, stride_vk, stride_vn,  #
                                    spi_row, stride_spin, sparsity_count,  #
                                    qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4, offs_n, offs_d, N_CTX_KV, True, fp8_v)

    # epilogue
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    l_ptrs = L + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX_Q)
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX_Q)
    # will get optimized out when Out_scale is 1.0 since it's tl.constexpr
    acc *= Out_scale
    o_accum_ptrs = Out_accum + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    acc += tl.load(o_accum_ptrs, mask=qo_mask)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=qo_mask)


class _sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sparsity_indices, sparsity_counts, O_scale=1.0, scale=None):
        o_accum = torch.zeros_like(q)
        sm_scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[1] >= k.shape[1] and q.shape[1] % k.shape[1] == 0
        assert k.shape[1] == v.shape[1]
        assert k.shape[2] == v.shape[2]
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)

        bm = get_kernel_config_attn()['bm']
        assert bm % 64 == 0, "BM must be a multiple of 64"
        num_qg_per_indices_group = bm // 64

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _sparse_attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, o_accum, O_scale,  #
            sparsity_indices, sparsity_counts,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            sparsity_indices.stride(0), sparsity_indices.stride(1), sparsity_indices.stride(2), sparsity_indices.stride(3),  #
            sparsity_counts.stride(0), sparsity_counts.stride(1), sparsity_counts.stride(2),  #
            q.shape[0], q.shape[1], k.shape[1],  #
            N_CTX_Q=q.shape[2],  #
            N_CTX_KV=k.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            num_qg_per_indices_group=num_qg_per_indices_group,
            fp8_v=(v.dtype == torch.float8_e5m2),
        )

        return o, (M.unsqueeze(-1), L.unsqueeze(-1))


# csp_attn = _sparse_attention.apply
def csp_attn(q, k, v, sparsity_indices, sparsity_counts, O_scale=1.0, scale=None):
    return _sparse_attention.apply(q, k, v, sparsity_indices, sparsity_counts, O_scale, scale)


def main():
    """Smoke test: full indices (all keys selected) must match dense SDPA."""
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    batch, num_heads, q_len, kv_len, head_dim = 2, 24, 1600, 6400, 128
    n_groups = (q_len + 192 - 1) // 192

    q = torch.randn(batch, num_heads, q_len, head_dim)
    k = torch.randn(batch, num_heads, kv_len, head_dim)
    v = torch.randn(batch, num_heads, kv_len, head_dim)

    indices = torch.arange(kv_len, dtype=torch.int32).repeat((batch, num_heads, n_groups, 1)).contiguous()
    counts = torch.full((batch, num_heads, n_groups), kv_len, dtype=torch.int32)

    o, _ = csp_attn(q, k, v, indices, counts, 1.0)
    o_ref = F.scaled_dot_product_attention(q, k, v)
    print(f"{o.shape=}, max_diff={(o - o_ref).abs().max():.3f}")


if __name__ == '__main__':
    main()
