import triton
import triton.language as tl
import torch
import math
import torch.nn.functional as F

DEVICE = 'cuda'


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    block_index_q, qk_scale,
                    seqlen,
                    stride_v_seqlen, stride_k_seqlen,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX_KV: tl.constexpr, fp8_v: tl.constexpr, should_mask_kv: tl.constexpr):
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX_KV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # k = tl.load(K_block_ptr)
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, k)

        # m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        # qk = qk * qk_scale - m_ij[:, None]

        # if should_mask_kv:
        #     qk = tl.where(start_n + offs_n[None, :] < N_CTX_KV, qk, -1.0e6)
        # -->
        valid_n = (start_n + offs_n[None, :]) < N_CTX_KV
        qk = tl.where(valid_n, qk, -float("inf"))

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
        # v = tl.load(V_block_ptr)
        v = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")
        
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    # for BM in [64, 128]\
    # for BN in [32, 64]\
    for BM in [64]\
    for BN in [64]\
    for s in ([3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["N_CTX_Q", "N_CTX_KV", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, L, Out, seqlen,  # seqlen is q_seqlen
              stride_q_batch, stride_q_head, stride_q_seqlen, stride_q_dim,  #
              stride_k_batch, stride_k_head, stride_k_seqlen, stride_k_dim,  #
              stride_v_batch, stride_v_head, stride_v_seqlen, stride_v_dim,  #
              stride_o_batch, stride_o_head, stride_o_seqlen, stride_o_dim,  #
              Z, H, # q batch, num_heads
              N_CTX_Q: tl.constexpr,
              N_CTX_KV: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,
              should_mask_kv: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // H
    index_head = index_batch_head % H

    qo_offset = index_batch.to(tl.int64) * stride_q_batch + index_head.to(tl.int64) * stride_q_head
    k_offset = index_batch.to(tl.int64) * stride_k_batch + index_head.to(tl.int64) * stride_k_head
    v_offset = index_batch.to(tl.int64) * stride_v_batch + index_head.to(tl.int64) * stride_v_head
    
    offs_m = block_index_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_headsize = tl.arange(0, HEAD_DIM)
    
    Q_block_ptr = (
        Q
        + qo_offset
        + offs_m[:, None] * stride_q_seqlen
        + offs_headsize[None, :] * stride_q_dim
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    # V_block_ptr = (
    #     V
    #     + v_offset
    #     + offs_n[:, None] * stride_vk
    #     + offs_headsize[None, :] * stride_vn
    # )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX_KV, HEAD_DIM),
        strides=(stride_v_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )

    # K_block_ptr = (
    #     K
    #     + k_offset
    #     + (offs_n[None, :] // BLOCK_N) * stride_kn
    #     + offs_headsize[:, None] * stride_kk
    # )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_ptrs = (
        Out 
        + qo_offset 
        + (block_index_q * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_o_seqlen 
        + tl.arange(0, HEAD_DIM)[None, :] * stride_o_dim
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # O_block

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504      # 1/log(2)
    qo_mask = (offs_m < N_CTX_Q)[:, None]
    q = tl.load(Q_block_ptr, mask=qo_mask)

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    block_index_q, qk_scale, seqlen,  #
                                    stride_v_seqlen, stride_k_seqlen,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, N_CTX_KV, V.dtype.element_ty == tl.float8e5, should_mask_kv)

    acc = acc / l_i[:, None]
    m_ptrs = M + index_batch_head * N_CTX_Q + offs_m
    l_ptrs = L + index_batch_head * N_CTX_Q + offs_m

    tl.store(m_ptrs, m_i, mask=offs_m < seqlen)
    tl.store(l_ptrs, l_i, mask=offs_m < seqlen)
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=qo_mask)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale=None):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        sm_scale = 1/math.sqrt(HEAD_DIM_K) if scale is None else scale

        stage = 1       # no causal
        should_mask_kv = k.shape[-2] % 64 != 0      # ? after models/infinity/sparse_attn_ops.py padding, is it useful?
        # extra_kern_args = {''}
        extra_kern_args = {'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 3, 'num_warps': 4}

        # Grid calculation is based on Query's sequence length, which is correct.
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        
        # M and L have the same sequence length as Q
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)     # 存储每个Query的logsumexp中的最大值m，用于数值稳定
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)     # 存储每个Query的logsumexp的指数部分l (即Σexp(z_i - m))
        seqlen = q.shape[2]

        _attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, seqlen,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],     # batch size, num heads
            N_CTX_Q=q.shape[2],         # q_len
            N_CTX_KV=k.shape[2],        # kv_len
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,  #
            should_mask_kv=should_mask_kv,
            **extra_kern_args)

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
    # print(torch.allclose(o, o_ref, atol=1e-1, rtol=1e-1))
    print(torch.allclose(o, o_ref))


def main_cross_attn():
    """
    Test on an arbitrary sequence length that % 64 != 0.
    """
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    q_shape = (1, 24, 1600, 128)
    kv_shape = (1, 24, 4142, 128)
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
