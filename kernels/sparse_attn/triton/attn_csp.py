# import os
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
from torch.nn import functional as F
import math
import triton
import triton.language as tl
from einops import rearrange

from kernels.sparse_attn.sparse_attn_config import get_kernel_config_attn

DEVICE = 'cuda'

cdiv = lambda a, b: (a + b - 1) // b
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64]\
    for BN in [64]\
    for s in [3]\
    for w in [4]\
]
def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

@triton.jit
def _sparse_attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr_orig, V_block_ptr_orig,  #
                    start_m, qk_scale,  #
                    indices_group,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX_KV: tl.constexpr, fp8_v: tl.constexpr, #
                    stride_k_seqlen, stride_v_seqlen,  #
                    sparsity_indices_ptr, sparsity_counts_ptr, #
                    should_mask_kv: tl.constexpr,
                    ):
    # 加载当前Query块需要处理的Key的数量
    sparsity_count = tl.load(sparsity_counts_ptr + indices_group)
    # sparsity_count = tl.load(sparsity_counts_ptr + start_m)
    # sparsity_count = 0

    # 计算稀疏索引指针的起始位置
    sparsity_offsets = tl.arange(0, BLOCK_N)
    sparsity_indices_ptr += indices_group * N_CTX_KV + sparsity_offsets
    # sparsity_indices_ptr += start_m * N_CTX + sparsity_offsets
    n_iters = tl.cdiv(sparsity_count, BLOCK_N)

    # 主循环：迭代所有稀疏的Key/Value块
    cur_iter = 0
    # loop over k, v and update accumulator
    for start_n in range(0, sparsity_count, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        sparsity_indices = tl.load(sparsity_indices_ptr)
        # sparsity_indices = tl.arange(0, BLOCK_N) + start_n
        # sparsity_indices = tl.zeros_like(sparsity_indices)

        tl.device_assert(tl.max(sparsity_indices) < N_CTX_KV, "Sparsity index out of bounds for Key sequence.")
        tl.device_assert(tl.min(sparsity_indices) >= 0, "Sparsity index is negative.")

        # 根据稀疏索引计算K和V块的指针 (Gather操作)
        K_block_ptr = K_block_ptr_orig + (sparsity_indices[None, :]) * stride_k_seqlen
        V_block_ptr = V_block_ptr_orig + (sparsity_indices[:, None]) * stride_v_seqlen
        # Commented out lines are for when we use random sparsity counts, in production it's always a multiple of BLOCK_N = 64
        # K_block_ptr = K_block_ptr_orig + (sparsity_indices[None, :] % N_CTX) * stride_k_seqlen
        # V_block_ptr = V_block_ptr_orig + (sparsity_indices[:, None] % N_CTX) * stride_v_seqlen
        # is_valid_mask = sparsity_offsets < sparsity_count - start_n # shape (BLOCK_N,)

        # load KV block
        k = tl.load(K_block_ptr)

        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        # qk = qk * qk_scale + tl.where(is_valid_mask[None, :], 0, -1.0e6)
        # qk -= m_ij[:, None]
        qk = qk * qk_scale - m_ij[:, None] # use fused multiply add!
        if should_mask_kv:
            qk = tl.where(start_n + offs_n[None, :] < N_CTX_KV, qk, -1.0e6)
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

        # 移动到下一批稀疏索引
        sparsity_indices_ptr += BLOCK_N
        cur_iter += 1

    return acc, l_i, m_i

@triton.autotune(list(filter(keep, configs)), key=["N_CTX_Q", "HEAD_DIM"])
@triton.jit
def _sparse_attn_fwd(Q, K, V, sm_scale, M, L, Out, Out_accum, Out_scale: tl.constexpr, #
              sparsity_indices, sparsity_counts, #
              stride_q_batch, stride_q_head, stride_q_len, stride_q_dim,  #
              stride_k_batch, stride_k_head, stride_k_len, stride_k_dim,  #
              stride_v_batch, stride_v_head, stride_v_len, stride_v_dim,  #
              stride_o_batch, stride_o_head, stride_o_len, stride_o_dim,  #
              stride_spiz, stride_spih,  #
              stride_spcz, stride_spch,  #
              Z, H, 
              N_CTX_Q, N_CTX_KV, #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,
              should_mask_kv: tl.constexpr,  #
              num_qg_per_indices_group: tl.constexpr,
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qo_offset = off_z.to(tl.int64) * stride_q_batch + off_h.to(tl.int64) * stride_q_head
    k_offset = off_z.to(tl.int64) * stride_k_batch + off_h.to(tl.int64) * stride_k_head
    v_offset = off_z.to(tl.int64) * stride_v_batch + off_h.to(tl.int64) * stride_v_head

    spi_offset = off_z.to(tl.int64) * stride_spiz + off_h.to(tl.int64) * stride_spih
    spi_ptr = sparsity_indices + spi_offset
    spc_offset = off_z.to(tl.int64) * stride_spcz + off_h.to(tl.int64) * stride_spch
    spc_ptr = sparsity_counts + spc_offset

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_headsize = tl.arange(0, HEAD_DIM)

    indices_group = start_m // num_qg_per_indices_group

    # block pointers
    Q_block_ptr = (
        Q
        + qo_offset
        + offs_m[:, None] * stride_q_len
        + offs_headsize[None, :] * stride_q_dim
    )
    K_block_ptr = (
        K
        + k_offset
        + (offs_n[None, :] // BLOCK_N) * stride_k_len
        + offs_headsize[:, None] * stride_k_dim
    )
    V_block_ptr = (
        V
        + v_offset
        + (offs_n[:, None] // BLOCK_N) * stride_v_len
        + offs_headsize[None, :] * stride_v_dim
    )
    O_block_ptr = (
        Out
        + qo_offset
        + offs_m[:, None] * stride_o_len
        + offs_headsize[None, :] * stride_o_dim
    )
    O_accum_block_ptr = (
        Out_accum
        + qo_offset
        + offs_m[:, None] * stride_o_len
        + offs_headsize[None, :] * stride_o_dim
    )

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    qo_mask = (offs_m < N_CTX_Q)[:, None]
    q = tl.load(Q_block_ptr, mask=qo_mask)

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _sparse_attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _sparse_attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _sparse_attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  #
                                    indices_group,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, N_CTX_KV, V.dtype.element_ty == tl.float8e5,  #
                                    stride_k_len, stride_v_len, #
                                    spi_ptr, spc_ptr, #
                                    should_mask_kv,
                                    )

    # epilogue 归一化并写回结果
    # m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    l_ptrs = L + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX_Q)
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX_Q)

    acc *= Out_scale # will get optimized out when Out_scale is 1.0 since it's tl.constexpr
    acc += tl.load(O_accum_block_ptr, mask=qo_mask)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=qo_mask)


class _sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, sparsity_indices, sparsity_counts, O_scale = 1.0):     # sparsity_indices[2, num_heads, q_groups, k_len], sparsity_counts[2, num_heads, q_groups(per q_group has num_KVs)]
        o_accum = torch.zeros_like(q)
        sm_scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 1

        # Mask the last partial KV block when K length is not a multiple of 64
        should_mask_kv = k.shape[-2] % 64 != 0
        extra_kern_args = {}
        # extra_kern_args = {'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 3, 'num_warps': 4}     # w/o autotune

        bm = get_kernel_config_attn()['bm']     # 192
        assert bm % 64 == 0, "BM must be a multiple of 64"
        num_qg_per_indices_group = bm // 64

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _sparse_attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, o_accum, O_scale,  #
            sparsity_indices, sparsity_counts, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            sparsity_indices.stride(0), sparsity_indices.stride(1), #
            sparsity_counts.stride(0), sparsity_counts.stride(1), #
            q.shape[0], q.shape[1],  # batch, num_heads
            N_CTX_Q=q.shape[2], N_CTX_KV=k.shape[2], #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            should_mask_kv=should_mask_kv,
            num_qg_per_indices_group=num_qg_per_indices_group,
            **extra_kern_args
        )

        return o, (M.unsqueeze(-1), L.unsqueeze(-1))

# csp_attn = _sparse_attention.apply
def csp_attn(q, k, v, sparsity_indices, sparsity_counts, O_scale = 1.0, scale=None):
    return _sparse_attention.apply(q, k, v, scale, sparsity_indices, sparsity_counts, O_scale)


def main():
    import pickle
    import chipmunk
    q, k, v, inds, counts = pickle.load(open('tensors.pkl', 'rb'))
    inds = inds[:, :, :, :q.shape[2]].contiguous()
    if not torch.all(counts % 64 == 0):
        breakpoint()
    for b in range(inds.shape[0]):
        for h in range(0, inds.shape[1]):
            for m in range(0, inds.shape[2]):
                # inds[b,h,m,:] = torch.arange(inds.shape[3]-1, -1, -1)
                # inds[b,h,m,:] = torch.arange(0, inds.shape[3])
                relevant_indices = inds[b,h,m,:counts[b,h,m]]
                if not torch.all((relevant_indices >= 0) & (relevant_indices < q.shape[2])):
                    breakpoint()
                pass
    print('beginning kernel...')
    o = chipmunk.ops.csp_attn(q, k, v, inds, counts)
    print(o.shape)


def cross_attn_test():
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)

    q_shape = (2, 24, 1600, 128)
    kv_shape = (2, 24, 4142, 128)

    q = torch.randn(q_shape)
    k = torch.randn(kv_shape)
    v = torch.randn(kv_shape)

    num_q_groups = math.ceil(q_shape[2] / 64)
    num_selected_keys = 640

    indices = torch.randint(0, kv_shape[2], (q_shape[0], q_shape[1], num_q_groups, num_selected_keys), dtype=torch.int32)
    counts = torch.full((q_shape[0], q_shape[1], num_q_groups), num_selected_keys, dtype=torch.int32)

    o_init = torch.zeros_like(q)
    result_delta, _ = csp_attn(q, k, v, indices, counts, 1.0)

    print(f'{q.shape=}')
    print(f'{result_delta.shape=}\n')


def cross_attn_test2():
    # torch.set_default_device('cuda')
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.bfloat16)

    q_shape = (2, 24, 1600, 128)
    kv_shape = (2, 24, 4142, 128)

    q = torch.randn(q_shape)
    k = torch.randn(kv_shape)
    v = torch.randn(kv_shape)

    num_q_groups = math.ceil(q_shape[2] / 64)
    # num_selected_keys = 640

    all_indices = torch.arange(kv_shape[2], dtype=torch.int32).expand(q_shape[0], q_shape[1], num_q_groups, -1)
    all_counts = torch.full((q_shape[0], q_shape[1], num_q_groups), kv_shape[2], dtype=torch.int32)

    o_dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    o_sparse_full, _ = csp_attn(q, k, v, all_indices, all_counts, 1.0)

    print(f'{o_dense.shape=}')
    print(f'{o_sparse_full.shape=}')

    assert torch.allclose(o_sparse_full, o_dense, rtol=1e-2, atol=1e-2), "Cross-attention numerical mismatch with dense attention."
    breakpoint()


def cross_attn_test3():
    torch.set_default_device('cuda')
    # torch.set_default_dtype(torch.bfloat16)

    batch, num_heads, head_dim = 2, 24, 128
    # q_len, kv_len = 1600, 4142
    q_len, kv_len = 1600, 6400
    
    n_groups = (q_len + 192 - 1) // 192

    def make_tensor(shape, is_contiguous, fill_value=None):
        if is_contiguous:
            new_vec = torch.randn(shape)
        else:
            new_shape = (shape[2], shape[0], shape[1], shape[3])
            new_vec = torch.randn(*new_shape)
            new_vec = new_vec.permute(1, 2, 0, 3)
        if fill_value is not None:
            new_vec.fill_(fill_value)
        return new_vec
    
    is_contiguous = True
    q = make_tensor((batch, num_heads, q_len, head_dim), is_contiguous).to(torch.bfloat16)
    k = make_tensor((batch, num_heads, kv_len, head_dim), is_contiguous).to(torch.bfloat16)
    v = make_tensor((batch, num_heads, kv_len, head_dim), is_contiguous).to(torch.bfloat16)
    # o = make_tensor((batch, num_heads, q_len, head_dim), True, fill_value=0)

    indices = torch.arange(kv_len, dtype=torch.int32).repeat((batch, num_heads, n_groups, 1)).contiguous()
    counts = torch.full((batch, num_heads, n_groups), 640, dtype=torch.int32)

    o, _ = csp_attn(q, k, v, indices, counts, 1)
    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    o_max_diff = (o - o_ref).abs().max()
    o_mean_diff = (o - o_ref).abs().mean()

    print(f"is_contig={is_contiguous}, seqlen: {q_len} (% 192 = {q_len % 192}, % 112 = {q_len % 112}), \
          o_max_diff: {o_max_diff:.2f}, o_mean_diff: {o_mean_diff:.2f}")


def make_qkv(
    B: int, H: int, Nq: int, Nk: int, D: int,
    dtype=torch.bfloat16, device="cuda", seed: int = 0
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(B, H, Nq, D, dtype=dtype, device=device, generator=g)
    k = torch.randn(B, H, Nk, D, dtype=dtype, device=device, generator=g)
    v = torch.randn(B, H, Nk, D, dtype=dtype, device=device, generator=g)
    
    return q, k, v


def build_random_inds_counts(
    B: int, H: int, Nq: int, Nk: int, bm: int = 192,
    cap_multiple: int = 64, seed: int = 0,
    top_keys: float = 0.15
):
    """
    构造随机 indices / counts：
      - 组维 G = ceil(Nq / bm)
      - 对每个 (b,h,g) 生成随机 count (1..min(Nk, 5*64))，常常不是 64 的倍数
      - capacity = 向上取整到 cap_multiple（与 BLOCK_N 对齐）
      - indices[b,h,g,:count] 为 [0, Nk) 的随机不重复索引；其余填充任意（不会被用到）
    """
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    G = (Nq + bm - 1) // bm
    indices_pad_to = 1

    # indices_count = int(cap * round((top_keys * Nk) / cap_multiple))
    indices_count = 640

    # 生成 indices
    indices = torch.empty(B, H, G, indices_count, dtype=torch.int32, device="cuda")
    for b in range(B):
        for h in range(H):
            for gg in range(G):
                chosen = torch.randperm(Nk, device="cuda", generator=g)[:indices_count].to(torch.int32)
                indices[b, h, gg, :indices_count] = chosen

    counts = torch.full(
        (B, H, triton.cdiv(Nq, bm)), 
        indices_count, device='cuda', dtype=torch.int32
    )
    padding_amount = (Nk - indices_count + indices_pad_to - 1) // indices_pad_to * indices_pad_to
    indices = torch.cat(
        [indices, torch.empty((*counts.shape, padding_amount), device='cuda', dtype=torch.int32)], 
        dim=-1
    ).to(torch.int32)

    return indices, counts


def reference_sparse_o(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    indices: torch.Tensor, counts: torch.Tensor,
    bm: int = 192
):
    """
    稠密参考：只保留被选列贡献（对未选列置 -inf），
    然后做 softmax(QK^T/sqrt(d)) @ V
    """
    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    G = (Nq + bm - 1) // bm
    assert indices.shape[:3] == (B, H, G)

    # logits: (B,H,Nq,Nk)
    scale = 1.0 / math.sqrt(D)
    logits = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32)) * scale

    # 为每行（按组）构造列掩码：True=可见（被选列），False=不可见
    mask = torch.zeros(B, H, Nq, Nk, dtype=torch.bool, device=q.device)
    for b in range(B):
        for h in range(H):
            for gg in range(G):
                s = gg * bm
                e = min((gg + 1) * bm, Nq)
                c = int(counts[b, h, gg].item())
                sel = indices[b, h, gg, :c]  # (c,)
                # 该 group 内所有 query 行共享同一列集合
                mask[b, h, s:e, sel] = True

    # 对未选列置 -inf（或一个极小值）
    logits_masked = torch.where(mask, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
    P = torch.softmax(logits_masked, dim=-1)  # (B,H,Nq,Nk), fp32
    O = torch.matmul(P.to(v.dtype), v.to(v.dtype))
    return O


def tokens_per_sec(B, H, Nq, ms_per_iter):
    # 按“处理的 query tokens 数”估算吞吐
    total_tokens = B * H * Nq
    sec = ms_per_iter / 1000.0
    return total_tokens / sec


def csp_cross_attn_test(
    B: int = 2, H: int = 16, Nq: int = 4096, Nk: int = 10521, D: int = 128,
    bm: int = 192,
    dtype=torch.bfloat16,
    device="cuda",
    atol=2e-2, rtol=2e-2,
    seed: int = 0,
):
    q, k, v = make_qkv(B, H, Nq, Nk, D, dtype=dtype, device=device, seed=seed)
    print(f'{q.shape=}, {k.shape=}')
    indices, counts = build_random_inds_counts(B, H, Nq, Nk, bm=bm, seed=seed)
    # print(f'{indices.shape=}, {counts.shape=}')

    # 参考输出（只保留被选列）
    O_ref = reference_sparse_o(q, k, v, indices, counts, bm=bm)

    # 被测：Chipmunk 稀疏核（triton 包装）
    # 其 Python 包装会做：o = o + o_delta * o_scale
    # 为了得到纯 delta，我们传入全 0 的 o，并设 o_scale=1.0
    o_scale=1.0
    o_zero = torch.zeros_like(q)
    o_delta, _ = csp_attn(q, k, v, indices, counts, 1.0, scale=None)
    assert o_delta.shape == o_zero.shape, "Output delta shape mismatch - o_delta: {}, o: {}".format(o_delta.shape, o_zero.shape)
    o_out = o_zero + o_delta * o_scale

    ok = torch.allclose(o_out, O_ref, atol=atol, rtol=rtol)
    max_abs_err = (o_out - O_ref).abs().max().item()

    print(f'\nAccuracy: {ok=}, {max_abs_err=}')

    # ------ latency_test ------
    warmup=20; iters=50

    # ~~~~~~~~~~~~~~~~
    # Torch full-attn
    # ~~~~~~~~~~~~~~~~

    # warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # measure
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    tps_sparse = tokens_per_sec(B, H, Nq, ms)

    print(f'Torch SDPA: {ms} ms, tokens_per_sec: {tps_sparse}')

    # ~~~~~~~~
    # Triton
    # ~~~~~~~~
    # warmup
    for _ in range(warmup):
        _, _ = csp_attn(q, k, v, indices, counts, 1.0, scale=None)
    torch.cuda.synchronize()

    # measure
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _, _ = csp_attn(q, k, v, indices, counts, 1.0, scale=None)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    tps_sparse = tokens_per_sec(B, H, Nq, ms)

    print(f'Triton Kernel: {ms} ms, tokens_per_sec: {tps_sparse}')


if __name__ == '__main__':
    # main()
    # cross_attn_test()
    # cross_attn_test2()
    # cross_attn_test3()
    
    csp_cross_attn_test(Nq=2304, Nk=6425, seed=42)
    csp_cross_attn_test(seed=10086)
    
