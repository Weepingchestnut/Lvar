import os
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import math
import triton
import triton.language as tl

from torch.nn import functional as F


def check_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous, "A tensor is not contiguous"
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"


DEVICE = 'cuda'

cdiv = lambda a, b: (a + b - 1) // b

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
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

@triton.jit
def _full_attn_fwd_inner(acc, l_i, m_i, q,  #
                    blocksums_ptrs,
                    blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n,
                    stride_vk, stride_kn,
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale, 
                    q_len, kv_len, #
                    H, #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr,
                    should_mask_kv: tl.constexpr,
                    ):
    off_hb = tl.program_id(1)
    off_b = off_hb // H
    off_h = off_hb % H

    # blocksums_ptrs += off_b.to(tl.int64) * blocksums_stride_b + off_h.to(tl.int64) * blocksums_stride_h + start_m * blocksums_stride_m + tl.arange(0, BLOCK_N) * blocksums_stride_n
    bsp = (
        blocksums_ptrs 
        + off_b.to(tl.int64) * blocksums_stride_b 
        + off_h.to(tl.int64) * blocksums_stride_h 
        + (start_m // 3) * blocksums_stride_m 
        + tl.arange(0, BLOCK_N) * blocksums_stride_n
    )

    # blocksums = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # q_dot_k = tl.dot(q, k)
        q_dot_k = tl.dot(q, k)
        # q_dot_k = tl.where(start_n + offs_n[None, :] < 4592, q_dot_k, -1.0e6)
        qk = q_dot_k
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        # qk = qk * qk_scale - m_ij[:, None]
        qk = qk * qk_scale - m_ij[:, None]
        if should_mask_kv:
            qk = tl.where(start_n + offs_n[None, :] < kv_len, qk, -1.0e6)
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")
        # v = tl.where(start_n + offs_n[:, None] < 4592, v, 0).to(tl.bfloat16)
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

        # ----------------- UPDATE POINTERS -----------------
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
    return acc, l_i, m_i


@triton.autotune(list(filter(keep, configs)), key=["N_CTX_Q", "HEAD_DIM"])
@triton.jit
def _full_attn_fwd(Q, K, V, sm_scale, M, L, Out, 
                   q_len, kv_len, #
              blocksums_ptrs, #
              blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n, #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, 
              N_CTX_Q: tl.constexpr,  #
              N_CTX_KV: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              should_mask_kv: tl.constexpr,
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qo_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    offs_headsize = tl.arange(0, HEAD_DIM)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # block pointers
    Q_block_ptr = (
        Q
        + qo_offset
        + offs_m[:, None] * stride_qm
        + offs_headsize[None, :] * stride_qk
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)

    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    offs_o = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_on
    O_ptrs = Out + qo_offset + offs_o

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
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _full_attn_fwd_inner(
        acc, l_i, m_i, q, 
        blocksums_ptrs,
        blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n,
        stride_vk, stride_kn,
        K_block_ptr, V_block_ptr,  #
        start_m, qk_scale, 
        q_len, kv_len, #
        H, #
        BLOCK_M, HEAD_DIM, BLOCK_N,  #
        4 - STAGE, offs_m, offs_n, N_CTX_Q, V.dtype.element_ty == tl.float8e5, should_mask_kv  #
    )
    # epilogue
    # m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    l_ptrs = L + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < q_len)
    tl.store(l_ptrs, l_i, mask=offs_m < q_len)
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=qo_mask)

    # ---- 第二阶段：精确列和（不依赖 prev_lse）----
    # 重新构造 K/V block ptr（只用 K；V 不再需要）
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset, shape=(N_CTX_KV, HEAD_DIM), strides=(stride_vk, stride_vn),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset, shape=(HEAD_DIM, N_CTX_KV), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
    )
    # blocksums 写指针（与第一阶段相同的布局）
    bsp = (blocksums_ptrs
        + off_z.to(tl.int64) * blocksums_stride_b
        + off_h.to(tl.int64) * blocksums_stride_h
        + (start_m // 3) * blocksums_stride_m
        + tl.arange(0, BLOCK_N) * blocksums_stride_n
    )

    # 再扫一遍列：用最终 m_i, l_i 计算 p_norm 并对列求和
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        q_dot_k = tl.dot(q, k)                             # [BLOCK_M, BLOCK_N]
        qk = q_dot_k * qk_scale
        qk = qk - m_i[:, None]                             # 用最终 m_i
        if should_mask_kv:
            qk = tl.where(start_n + offs_n[None, :] < kv_len, qk, -1.0e6)
        p_norm = tl.math.exp2(qk) / l_i[:, None]           # 用最终 l_i 归一化
        colsum = tl.sum(p_norm, 0)                         # [BLOCK_N]
        tl.atomic_add(
            bsp, colsum,
            mask=(start_n + offs_n) < kv_len, 
            sem='relaxed'
        )
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        bsp += BLOCK_N * blocksums_stride_n


class _full_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale=None):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        sm_scale = 1/math.sqrt(HEAD_DIM_K) if scale is None else scale
        stage = 1
        should_mask_kv = k.shape[-2] % 64 != 0
        # extra_kern_args = {'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 3, 'num_warps': 4}
        # mb = q.shape[2] // 128 if q.shape[2] % 128 == 0 else q.shape[2] // 128 + 1
        # fuse_amt = get_kernel_config_attn()['bm'] // 64
        mb = triton.cdiv(q.shape[2], 192)
        # mb = ((mb + fuse_amt - 1) // fuse_amt) * fuse_amt  # Round up to nearest multiple of 3
        # breakpoint()
        # print(f'mb: {mb}')
        # print(f'q.shape[2] // 128: {q.shape[2] // 128}')
        # print(f'triton cdiv: {triton.cdiv(q.shape[2], 128)}')

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        # grid = lambda args: (mb, q.shape[0] * q.shape[1], 1)

        # print(f'grid: {grid({})}')
        
        M = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.zeros_like(M, dtype=torch.float32)
        o = torch.empty_like(q)
        
        blocksums = torch.zeros((q.shape[0], q.shape[1], mb, k.shape[2]), device=q.device, dtype=torch.float32)
        # print(f'blocksums: {blocksums.shape}')
        q_len = q.shape[2]
        kv_len = k.shape[2]
        # seqlen = 4592
        _full_attn_fwd[grid](
            q, k, v, sm_scale, M, L, o, 
            q_len, kv_len, #
            blocksums,
            blocksums.stride(0), blocksums.stride(1), blocksums.stride(2), blocksums.stride(3), #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX_Q=q.shape[2],  #
            N_CTX_KV=k.shape[2],
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,
            should_mask_kv=should_mask_kv,
            # **extra_kern_args,
        )

        return o, blocksums, (M.unsqueeze(-1), L.unsqueeze(-1))

# dense_colsum_attn = _full_attention.apply
def dense_colsum_attn(q, k, v, scale=None):
    return _full_attention.apply(q, k, v, scale)


def main():
    from attn_dense import dense_attn
    from tqdm import tqdm
    import pickle
    """
    Test on an arbitrary sequence length that % 64 != 0.
    """
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)
    # pkl = pickle.load(open('tensors.pkl', 'rb'))

    for i in tqdm(range(1)):
        qkv_shape = (1, 24, 67324, 128)
        q = torch.randn(qkv_shape)
        k = torch.randn(qkv_shape)
        v = torch.randn(qkv_shape)

        # _, prev_lse = dense_attn(q, k, v)
        
        # q,k,v,prev_lse = pkl
        o, blocksums, new_lse = dense_colsum_attn(q, k, v)
        o_ref = F.scaled_dot_product_attention(q, k, v)

        assert torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2), "Dense Colsum Attention is not close to output ref"
        # breakpoint()


def main_cross_attn():
    from attn_dense import dense_attn
    from tqdm import tqdm
    import pickle
    """
    Test on an arbitrary sequence length that % 64 != 0.
    """
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.bfloat16)
    # pkl = pickle.load(open('tensors.pkl', 'rb'))
    pad_to = 192

    for i in tqdm(range(1)):
        q_shape = (1, 24, 1600, 128)
        kv_shape = (1, 24, 4142, 128)
        q = torch.randn(q_shape)
        k = torch.randn(kv_shape)
        v = torch.randn(kv_shape)
        
        # q,k,v,prev_lse = pkl
        # o, blocksums, new_lse = dense_colsum_attn(q, pad_qkvo_tensor(k, pad_to), pad_qkvo_tensor(v, pad_to))
        o, blocksums, new_lse = dense_colsum_attn(q, k, v)
        o_ref = F.scaled_dot_product_attention(q, k, v)

        assert torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2), "Dense Colsum Attention is not close to output ref"
        # breakpoint()


def test_dense_colsum_attn_cross_shapes_and_values():
    BM = 192
    torch.manual_seed(0)

    device = "cuda"
    dtype  = torch.bfloat16  # 与内核默认保持一致；也可切成 float16/float32 试试

    # ---------- 造一个 cross-attn 形状 ----------
    B, H, D = 2, 3, 64                  # D ∈ {16,32,64,128,256}
    Nq = 777                            # 非整除 BLOCK_M 的行长度
    Nk = 1234                           # 非整除 BLOCK_N 的列长度（刻意不整除）
    q = torch.randn(B, H, Nq, D, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(B, H, Nk, D, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(B, H, Nk, D, device=device, dtype=dtype, requires_grad=False)

    # ---------- 构造“上一轮 LSE 常数” prev_lse ----------
    # 计算 S = Q K^T / sqrt(D)
    scale = 1.0 / math.sqrt(D)
    # 为了减少内存峰值，分块计算（也可一次性算）
    # 这里直接一次性计算即可，Nq*Nk 不算太大
    S = torch.einsum("b h m d, b h n d -> b h m n", q.to(torch.float32), k.to(torch.float32)) * scale
    # 行向：m over keys
    prev_maxes = S.amax(dim=-1, keepdim=True)                      # (B,H,Nq,1)
    prev_norm  = torch.exp2((S - prev_maxes) * (1.0 / math.log(2))).sum(dim=-1, keepdim=True)  # log base-2 的一致性可选
    # 为了和 kernel 内的基数一致，直接用自然底也可以（两边一致即可）
    # 这里直接采用自然底做法（与前式等价，只要两边一致）：
    prev_maxes_e = S.amax(dim=-1, keepdim=True)                    # (B,H,Nq,1)
    prev_norm_e  = torch.exp(S - prev_maxes_e).sum(dim=-1, keepdim=True)

    prev_lse = (prev_maxes_e, prev_norm_e)  # 作为 dense_colsum_attn 的 p（triton 分支要求 tuple）

    # ---------- 调用被测算子 ----------
    o_kernel, cs_kernel, l_kernel = dense_colsum_attn(q, k, v, prev_lse)

    # ---------- 参考实现 1：前向输出 o 的数值对比 ----------
    # 参考：o_ref = softmax(S) @ V
    P_ref = torch.softmax(S, dim=-1)                                 # (B,H,Nq,Nk)
    o_ref = torch.einsum("b h m n, b h n d -> b h m d", P_ref, v.to(torch.float32)).to(o_kernel.dtype)
    # 误差阈值：bfloat16 下建议宽松一点
    assert torch.allclose(o_kernel, o_ref, atol=2e-2, rtol=2e-2), \
        f"o mismatch: max abs err={ (o_kernel - o_ref).abs().max().item():.6f}"

    # ---------- 参考实现 2：列和矩阵 cs 的数值对比 ----------
    # kernel 的 cs 形状为 (B,H, ceil(Nq/BM), Nk)
    G = (Nq + BM - 1) // BM
    assert cs_kernel.shape == (B, H, G, Nk), f"cs shape mismatch: got {cs_kernel.shape}, expect {(B,H,G,Nk)}"

    # 用“上一轮 lse 常数”重建 p_prev，并在 query-group 上求和
    # p_prev[m, n] = exp(S[m,n] - m_row) / l_row
    P_prev = torch.exp(S - prev_maxes_e) / prev_norm_e              # (B,H,Nq,Nk)

    cs_ref = torch.zeros((B, H, G, Nk), device=device, dtype=P_prev.dtype)
    for g in range(G):
        m_start = g * BM
        m_end   = min((g + 1) * BM, Nq)
        # sum over rows in this query-group
        cs_ref[:, :, g, :] = P_prev[:, :, m_start:m_end, :].sum(dim=2)

    cs_ref = cs_ref.to(cs_kernel.dtype)
    # 列和更容易受 bf16 的舍入影响，阈值略放宽
    assert torch.allclose(cs_kernel, cs_ref, atol=3e-2, rtol=3e-2), \
        f"cs mismatch: max abs err={ (cs_kernel - cs_ref).abs().max().item():.6f}"

    # ---------- 参考实现 3：L 的形状/数值（可选，仅形状断言更稳妥） ----------
    # l_kernel 依实现可能是 (M,L) 或 (M,L) tuple；这里仅断言行维度对齐
    if isinstance(l_kernel, tuple):
        for lk in l_kernel:
            assert lk.shape == (B, H, Nq, 1), f"L tuple item shape mismatch: {lk.shape}"
    else:
        assert l_kernel.shape == (B, H, Nq, 1), f"L shape mismatch: {l_kernel.shape}"


def test_dense_colsum_attn_cross_various_sizes(Nq, Nk, D):
    BM = 192
    torch.manual_seed(1)
    device, dtype = "cuda", torch.bfloat16
    B, H = 1, 2

    q = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
    k = torch.randn(B, H, Nk, D, device=device, dtype=dtype)
    v = torch.randn(B, H, Nk, D, device=device, dtype=dtype)

    scale = 1.0 / math.sqrt(D)
    S = torch.einsum("b h m d, b h n d -> b h m n", q.to(torch.float32), k.to(torch.float32)) * scale
    prev_maxes = S.amax(dim=-1, keepdim=True)
    prev_norm  = torch.exp(S - prev_maxes).sum(dim=-1, keepdim=True)
    prev_lse   = (prev_maxes, prev_norm)

    o_kernel, cs_kernel, l_kernel = dense_colsum_attn(q, k, v, prev_lse)

    # 形状断言
    assert o_kernel.shape == q.shape
    G = (Nq + BM - 1) // BM
    assert cs_kernel.shape == (B, H, G, Nk)

    # 简要数值 sanity：与 SDPA 的 o 接近
    P_ref = torch.softmax(S, dim=-1)
    o_ref = torch.einsum("b h m n, b h n d -> b h m d", P_ref, v.to(torch.float32)).to(o_kernel.dtype)
    assert torch.allclose(o_kernel, o_ref, atol=2e-2, rtol=2e-2)


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


def dense_reference_o_and_p(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    参考实现：
      P = softmax(QK^T / sqrt(d))
      O = P @ V
    返回 (O_ref, P_ref)；为数值稳定，内部用 fp32 计算 softmax。
    """
    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)

    # logits: (B,H,Nq,Nk)
    logits = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32)) * scale
    # Softmax over key axis
    P = torch.softmax(logits, dim=-1)  # (B,H,Nq,Nk), fp32
    # O = P @ V
    O = torch.matmul(P.to(v.dtype), v.to(v.dtype))  # 精度对齐到 v.dtype
    return O, P


def reference_blocksums(P: torch.Tensor, bm: int = 192):
    """
    把 P 的行（query 维）按 bm 分组，对每个 group 求列和：
      Bs[b,h,g,:] = sum_{i in group g} P[b,h,i,:]
    形状: (B,H,ceil(Nq/bm), Nk)
    """
    B, H, Nq, Nk = P.shape
    G = (Nq + bm - 1) // bm
    Bs = torch.zeros(B, H, G, Nk, dtype=P.dtype, device=P.device)
    for g in range(G):
        s = g * bm
        e = min((g + 1) * bm, Nq)
        Bs[:, :, g, :] = P[:, :, s:e, :].sum(dim=-2)  # sum over rows in group
    return Bs



def run_case(
    B: int, H: int, Nq: int, Nk: int, D: int,
    bm: int = 192,
    dtype=torch.bfloat16,
    device="cuda",
    atol_o=2e-2, rtol_o=2e-2,
    atol_cs=2e-2, rtol_cs=2e-2,
    seed: int = 0,
):
    q, k, v = make_qkv(B, H, Nq, Nk, D, dtype=dtype, device=device, seed=seed)

    # 参考输出与概率
    O_ref, P_ref = dense_reference_o_and_p(q, k, v)

    # 被测实现（Triton 路径忽略 p）
    o, cs, l = dense_colsum_attn(q, k, v)

    # 1) 检查 O
    assert o.shape == O_ref.shape
    ok_o = torch.allclose(o, O_ref, atol=atol_o, rtol=rtol_o)

    # 2) 参考列和
    Bs_ref = reference_blocksums(P_ref, bm=bm)
    assert cs.shape == Bs_ref.shape, f"cs.shape={cs.shape}, Bs_ref.shape={Bs_ref.shape}"
    ok_cs = torch.allclose(cs, Bs_ref, atol=atol_cs, rtol=rtol_cs)

    return ok_o, ok_cs, {
        "max_abs_err_o": (o - O_ref).abs().max().item(),
        "max_abs_err_cs": (cs - Bs_ref).abs().max().item(),
        "shape_o": tuple(o.shape),
        "shape_cs": tuple(cs.shape),
    }


def test_dense_colsum_vs_reference(B, H, Nq, Nk, D):
    # if not torch.cuda.is_available():
    #     pytest.skip("CUDA required")
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    ok_o, ok_cs, stats = run_case(
        B, H, Nq, Nk, D,
        bm=192,
        dtype=dtype, device=device,
        atol_o=2e-2, rtol_o=2e-2,
        atol_cs=2e-2, rtol_cs=2e-2,
        seed=42,
    )

    print(f"[B{B} H{H} Nq{Nq} Nk{Nk} D{D}] -> "
          f"O ok={ok_o}, CS ok={ok_cs}, "
          f"max_abs_err_o={stats['max_abs_err_o']:.4f}, "
          f"max_abs_err_cs={stats['max_abs_err_cs']:.4f}, "
          f"shape_o={stats['shape_o']}, shape_cs={stats['shape_cs']}")

    assert ok_o, f"O mismatch: max_abs_err={stats['max_abs_err_o']}"
    assert ok_cs, f"CS mismatch: max_abs_err={stats['max_abs_err_cs']}"


if __name__ == '__main__':
    main()
    main_cross_attn()

    # test_dense_colsum_attn_cross_shapes_and_values()

    # test_dense_colsum_attn_cross_various_sizes(512, 640, 64)
    # test_dense_colsum_attn_cross_various_sizes(513, 1000, 128)
    # test_dense_colsum_attn_cross_various_sizes(777, 1234, 64)

    test_dense_colsum_vs_reference(1, 4, 256, 1000, 64)

