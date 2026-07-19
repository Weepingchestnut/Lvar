import triton
import triton.language as tl
import torch
import math
import torch.nn.functional as F

DEVICE = 'cuda'


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K, V, k_offset, v_offset,  #
                    stride_kn, stride_kk, stride_vk, stride_vn,  #
                    qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_n: tl.constexpr, offs_d: tl.constexpr,  #
                    N_CTX_KV: tl.constexpr, MASK_KV: tl.constexpr, fp8_v: tl.constexpr):
    # loop-split: STAGE 3 sweeps the KV blocks that fit fully inside N_CTX_KV (no
    # boundary mask); STAGE 4 handles the trailing partial block (masked).
    if STAGE == 3:
        lo, hi = 0, (N_CTX_KV // BLOCK_N) * BLOCK_N
    else:
        lo, hi = (N_CTX_KV // BLOCK_N) * BLOCK_N, N_CTX_KV
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k_ptrs = K + k_offset + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kk
        if MASK_KV:
            kv_mask = start_n + offs_n < N_CTX_KV
            k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)
        else:
            k = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        if MASK_KV:
            qk = tl.where(kv_mask[None, :], qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
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
    return acc, l_i, m_i


# Autotune space mirrors fused_attention.py (dense, no query-group constraint here,
# so BLOCK_M can grow to 128). N_CTX_KV is intentionally kept out of the key: it
# tracks k_len, which varies per video (prompt length); keeping it re-autotunes
# every video and the OutOfResources exceptions raised while benchmarking retain
# (via their tracebacks) the decision-scale activations -> per-video GPU leak.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
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


@triton.autotune(list(filter(keep, configs)), key=["N_CTX_Q", "HEAD_DIM"],
                 prune_configs_by={'early_config_prune': prune_block_n})
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, L, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H_Q, H_KV, N_CTX_Q, N_CTX_KV,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              fp8_v: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h = off_hz % H_Q
    # Map Q heads to KV heads for cross-attention/GQA. H_Q == H_KV keeps
    # the original self-attention address arithmetic.
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

    # full KV blocks (unmasked) then the trailing partial block (masked)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                    K, V, k_offset, v_offset,  #
                                    stride_kn, stride_kk, stride_vk, stride_vn,  #
                                    qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    3, offs_n, offs_d, N_CTX_KV, False, fp8_v)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                    K, V, k_offset, v_offset,  #
                                    stride_kn, stride_kk, stride_vk, stride_vn,  #
                                    qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4, offs_n, offs_d, N_CTX_KV, True, fp8_v)

    # epilogue: keep the running max (M) and sum (L) separate -- downstream
    # colsum/sparse passes consume prev_max and prev_norm individually, so do NOT
    # fold them into a combined log-sum-exp here.
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    l_ptrs = L + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX_Q)
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX_Q)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=qo_mask)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale=None):
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
        sm_scale = 1 / math.sqrt(HEAD_DIM_K) if scale is None else scale
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, sm_scale, M, L, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], k.shape[1],  #
            N_CTX_Q=q.shape[2],  #
            N_CTX_KV=k.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            fp8_v=(v.dtype == torch.float8_e5m2),  #
        )

        return o, (M.unsqueeze(-1), L.unsqueeze(-1))


# dense_attn = _attention.apply
def dense_attn(q, k, v, scale=None):
    return _attention.apply(q, k, v, scale)


def main():
    """
    Test on an arbitrary sequence length that % 64 != 0.
    """
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    qkv_shape = (1, 24, 2385, 128)
    q = torch.randn(qkv_shape)
    k = torch.randn(qkv_shape)
    v = torch.randn(qkv_shape)
    o, (M, L) = dense_attn(q, k, v)
    o_ref = F.scaled_dot_product_attention(q, k, v)
    print(o.shape, o_ref.shape)
    print(torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2))


def main_cross_attn():
    """
    Test on an arbitrary sequence length that % 64 != 0.
    """
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    q_shape = (1, 24, 4096, 128)
    kv_shape = (1, 24, 10521, 128)
    q = torch.randn(q_shape)
    k = torch.randn(kv_shape)
    v = torch.randn(kv_shape)
    o, (M, L) = dense_attn(q, k, v)
    o_ref = F.scaled_dot_product_attention(q, k, v)
    print(o.shape, o_ref.shape)
    print(torch.allclose(o, o_ref, atol=1e-1, rtol=1e-1))


if __name__ == '__main__':
    # main()
    main_cross_attn()
