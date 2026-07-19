import os
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import math
import triton
import triton.language as tl

from torch.nn import functional as F

from .attn_dense import dense_attn


DEVICE = 'cuda'

cdiv = lambda a, b: (a + b - 1) // b

# BLOCK_M is pinned to 64: GROUP_TILES consecutive 64-row programs form one query
# group of group_rows = GROUP_TILES * 64 rows (start_m // GROUP_TILES; default 192),
# so only BLOCK_N / num_stages / num_warps are tuned here.
configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BN in [32, 64, 128]\
    for s in [2, 3, 4]\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


def prune_block_n(configs, named_args, **kwargs):
    # BLOCK_N must satisfy the kernel's tl.static_assert(BLOCK_N <= HEAD_DIM).
    head_dim = kwargs["HEAD_DIM"]
    return [conf for conf in configs if conf.kwargs["BLOCK_N"] <= head_dim]


@triton.jit
def _full_attn_fwd_inner(acc, l_i, m_i, q,  #
                    prev_maxes_final, prev_normalization_final,  #
                    bsp_base, blocksums_stride_n,  #
                    K, V, k_offset, v_offset,  #
                    stride_kn, stride_kk, stride_vk, stride_vn,  #
                    qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr, offs_d: tl.constexpr,  #
                    N_CTX_Q: tl.constexpr, N_CTX_KV: tl.constexpr, MASK_KV: tl.constexpr,  #
                    should_mask_q: tl.constexpr, fp8_v: tl.constexpr):
    # loop-split: STAGE 3 sweeps full KV blocks (no boundary mask); STAGE 4 the
    # trailing partial block (masked).
    if STAGE == 3:
        lo, hi = 0, (N_CTX_KV // BLOCK_N) * BLOCK_N
    else:
        lo, hi = (N_CTX_KV // BLOCK_N) * BLOCK_N, N_CTX_KV
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k_ptrs = K + k_offset + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kk
        if MASK_KV:
            kv_mask = start_n + offs_n < N_CTX_KV
            k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)
        else:
            k = tl.load(k_ptrs)
        q_dot_k = tl.dot(q, k)
        if MASK_KV:
            q_dot_k = tl.where(kv_mask[None, :], q_dot_k, -1.0e6)

        # ---------------- CURRENT PHASE OF SOFTMAX (output o) -------------------
        m_ij = tl.maximum(m_i, tl.max(q_dot_k, 1) * qk_scale)
        qk = q_dot_k * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v_ptrs = V + v_offset + (start_n + offs_n)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        if MASK_KV:
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        else:
            v = tl.load(v_ptrs)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

        # ---------------- PREVIOUS PHASE OF SOFTMAX (colsum) -------------------
        qk_prev = q_dot_k * qk_scale - prev_maxes_final[:, None]
        p_prev = tl.math.exp2(qk_prev) / prev_normalization_final[:, None]
        if should_mask_q:
            p_prev = tl.where(offs_m[:, None] < N_CTX_Q, p_prev, 0.0)
        if MASK_KV:
            p_prev = tl.where(kv_mask[None, :], p_prev, 0.0)
        blocksums = tl.sum(p_prev, 0)
        bsp = bsp_base + (start_n + offs_n) * blocksums_stride_n
        if MASK_KV:
            tl.atomic_add(bsp, blocksums, mask=kv_mask, sem='relaxed')
        else:
            tl.atomic_add(bsp, blocksums, sem='relaxed')

    return acc, l_i, m_i


# blocksums is updated via atomic_add. Triton autotune benchmarks several configs
# on the same output tensor, so it must reset this accumulator between trials or
# the measured run will leave colsum multiplied by the tuning repeats.
# NOTE: N_CTX_KV dropped from the autotune key — it tracks per-video k_len and
# re-autotuning every video leaks the decision-scale activations via retained
# OutOfResources tracebacks. Reuse the config tuned on fixed N_CTX_Q/HEAD_DIM.
@triton.autotune(list(filter(keep, configs)), key=["N_CTX_Q", "HEAD_DIM"], reset_to_zero=["blocksums_ptrs"],
                 prune_configs_by={'early_config_prune': prune_block_n})
@triton.jit
def _full_attn_fwd(Q, K, V, sm_scale, M, L, Out,  #
              prev_maxes_ptr, prev_normalization_ptr,  #
              blocksums_ptrs,  #
              softmax_stride_b, softmax_stride_h, softmax_stride_n,  #
              blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H_Q, H_KV, N_CTX_Q, N_CTX_KV,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              GROUP_TILES: tl.constexpr,  #
              should_mask_q: tl.constexpr,  #
              fp8_v: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h = off_hz % H_Q
    # GQA mapping: each KV head serves H_Q / H_KV consecutive Q heads.
    off_h_kv = off_h // (H_Q // H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h_kv.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h_kv.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    qo_mask = (offs_m < N_CTX_Q)[:, None]
    q = tl.load(q_ptrs, mask=qo_mask, other=0.0)

    # previous-round softmax statistics (per query row), shared by the loop
    softmax_offset = off_z.to(tl.int64) * softmax_stride_b + off_h.to(tl.int64) * softmax_stride_h + offs_m * softmax_stride_n
    prev_maxes_final = tl.load(prev_maxes_ptr + softmax_offset, mask=(offs_m < N_CTX_Q), other=-1.0e6)
    prev_normalization_final = tl.load(prev_normalization_ptr + softmax_offset, mask=(offs_m < N_CTX_Q), other=1.0e6)

    # blocksums base: GROUP_TILES consecutive 64-row programs share one query group.
    bsp_base = (blocksums_ptrs
                + off_z.to(tl.int64) * blocksums_stride_b
                + off_h.to(tl.int64) * blocksums_stride_h
                + (start_m // GROUP_TILES).to(tl.int64) * blocksums_stride_m)

    # full KV blocks (unmasked) then the trailing partial block (masked)
    acc, l_i, m_i = _full_attn_fwd_inner(acc, l_i, m_i, q,  #
                                    prev_maxes_final, prev_normalization_final,  #
                                    bsp_base, blocksums_stride_n,  #
                                    K, V, k_offset, v_offset,  #
                                    stride_kn, stride_kk, stride_vk, stride_vn,  #
                                    qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    3, offs_m, offs_n, offs_d, N_CTX_Q, N_CTX_KV, False, should_mask_q, fp8_v)
    acc, l_i, m_i = _full_attn_fwd_inner(acc, l_i, m_i, q,  #
                                    prev_maxes_final, prev_normalization_final,  #
                                    bsp_base, blocksums_stride_n,  #
                                    K, V, k_offset, v_offset,  #
                                    stride_kn, stride_kk, stride_vk, stride_vn,  #
                                    qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4, offs_m, offs_n, offs_d, N_CTX_Q, N_CTX_KV, True, should_mask_q, fp8_v)

    # epilogue: keep current-round max (M) and sum (L) separate (next round's prev_lse)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    l_ptrs = L + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX_Q)
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX_Q)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=qo_mask)


class _full_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, prev_lse=None, scale=None, group_rows=192):
        sm_scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
        # When no previous-round lse is supplied, derive it from this q/k/v so the
        # colsum reflects the true softmax (standalone use / testing).
        if prev_lse is None:
            _, prev_lse = dense_attn(q, k, v, scale=sm_scale)
        prev_maxes, prev_normalization = prev_lse[0].squeeze(-1), prev_lse[1].squeeze(-1)
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
        should_mask_q = q.shape[-2] % 64 != 0
        # query-group rows for the colsum output; must be a multiple of the pinned
        # BLOCK_M=64 tile (GROUP_TILES programs share one blocksums row via atomic_add)
        group_rows = int(group_rows)
        assert group_rows > 0 and group_rows % 64 == 0, \
            f'colsum group_rows must be a positive multiple of 64, got {group_rows}'
        mb = triton.cdiv(q.shape[2], group_rows)

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        M = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.zeros_like(M, dtype=torch.float32)
        o = torch.empty_like(q)
        blocksums = torch.zeros((q.shape[0], q.shape[1], mb, k.shape[2]), device=q.device, dtype=torch.float32)

        _full_attn_fwd[grid](
            q, k, v, sm_scale, M, L, o,  #
            prev_maxes,  #
            prev_normalization,  #
            blocksums,
            prev_maxes.stride(0), prev_maxes.stride(1), prev_maxes.stride(2),  #
            blocksums.stride(0), blocksums.stride(1), blocksums.stride(2), blocksums.stride(3),  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], k.shape[1],  #
            N_CTX_Q=q.shape[2],  #
            N_CTX_KV=k.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            GROUP_TILES=group_rows // 64,
            should_mask_q=should_mask_q,
            fp8_v=(v.dtype == torch.float8_e5m2),
        )

        return o, blocksums, (M.unsqueeze(-1), L.unsqueeze(-1))


# dense_colsum_attn = _full_attention.apply
def dense_colsum_attn(q, k, v, prev_lse=None, scale=None, group_rows=192):
    return _full_attention.apply(q, k, v, prev_lse, scale, group_rows)


def main():
    """
    Smoke test on an arbitrary sequence length that % 64 != 0.
    """
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    qkv_shape = (1, 24, 2385, 128)
    q = torch.randn(qkv_shape)
    k = torch.randn(qkv_shape)
    v = torch.randn(qkv_shape)
    o, blocksums, new_lse = dense_colsum_attn(q, k, v)
    o_ref = F.scaled_dot_product_attention(q, k, v)
    print(o.shape, blocksums.shape)
    assert torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2), "Dense Colsum Attention output not close to ref"


if __name__ == '__main__':
    main()
