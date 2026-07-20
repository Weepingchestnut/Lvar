"""QK^T-only exact softmax column-sums, normalized by a caller-supplied row logsumexp.

Companion to attn_dense_colsum.py for the SparVAR decision step. The fused
kernel there re-runs a full online-softmax attention (V loads + PV dot + o
write) just to piggyback the colsum accumulation; when the caller already has
the attention output AND the per-row logsumexp (e.g. as a free byproduct of an
FA2 dense pass), the colsum only needs one more QK^T sweep:

    p[i, j] = exp2(q_i @ k_j * scale * log2(e) - lse2_i) == softmax(s)[i, j]

where ``lse2 = natural-log row logsumexp * log2(e)`` (the exp2-domain combined
logsumexp; equals M + log2(L) of attn_dense.py's split statistics).

Layout/contract matches dense_colsum_attn: q [Z, H_Q, N_CTX_Q, D],
k [Z, H_KV, N_CTX_KV, D] (GQA-native, H_Q % H_KV == 0), blocksums
[Z, H_Q, ceil(N_CTX_Q / group_rows), N_CTX_KV] fp32, accumulated via
atomic_add by GROUP_TILES consecutive 64-row programs per query group.
"""

import math

import torch
import triton
import triton.language as tl

cdiv = lambda a, b: (a + b - 1) // b

# BLOCK_M is pinned to 64 for the same reason as attn_dense_colsum.py: GROUP_TILES
# consecutive 64-row programs form one query group of group_rows = GROUP_TILES * 64.
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


@triton.jit
def _colsum_only_inner(q, lse2,  #
                       bsp_base, blocksums_stride_n,  #
                       K, k_offset, stride_kn, stride_kk,  #
                       qk_scale,  #
                       BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                       STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr, offs_d: tl.constexpr,  #
                       N_CTX_Q: tl.constexpr, N_CTX_KV: tl.constexpr, MASK_KV: tl.constexpr,  #
                       should_mask_q: tl.constexpr):
    # loop-split: STAGE 3 sweeps full KV blocks (no boundary mask); STAGE 4 the
    # trailing partial block (masked).
    if STAGE == 3:
        lo, hi = 0, (N_CTX_KV // BLOCK_N) * BLOCK_N
    else:
        lo, hi = (N_CTX_KV // BLOCK_N) * BLOCK_N, N_CTX_KV
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kk
        if MASK_KV:
            kv_mask = start_n + offs_n < N_CTX_KV
            k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)
        else:
            k = tl.load(k_ptrs)
        s2 = tl.dot(q, k) * qk_scale
        # exact softmax prob w.r.t. the supplied row logsumexp
        p = tl.math.exp2(s2 - lse2[:, None])
        if should_mask_q:
            p = tl.where(offs_m[:, None] < N_CTX_Q, p, 0.0)
        if MASK_KV:
            p = tl.where(kv_mask[None, :], p, 0.0)
        colsums = tl.sum(p, 0)
        bsp = bsp_base + (start_n + offs_n) * blocksums_stride_n
        if MASK_KV:
            tl.atomic_add(bsp, colsums, mask=kv_mask, sem='relaxed')
        else:
            tl.atomic_add(bsp, colsums, sem='relaxed')


# blocksums is accumulated via atomic_add, so autotune must reset it between
# trials (same as attn_dense_colsum.py). N_CTX_KV is likewise dropped from the
# autotune key: it tracks per-video k_len and re-autotuning every video would
# churn; reuse the config tuned on fixed N_CTX_Q/HEAD_DIM.
@triton.autotune(list(filter(keep, configs)), key=["N_CTX_Q", "HEAD_DIM"], reset_to_zero=["blocksums_ptrs"])
@triton.jit
def _colsum_only_fwd(Q, K, sm_scale, Lse2,  #
                     blocksums_ptrs,  #
                     lse_stride_b, lse_stride_h, lse_stride_n,  #
                     blocksums_stride_b, blocksums_stride_h, blocksums_stride_m, blocksums_stride_n,  #
                     stride_qz, stride_qh, stride_qm, stride_qk,  #
                     stride_kz, stride_kh, stride_kn, stride_kk,  #
                     Z, H_Q, H_KV, N_CTX_Q, N_CTX_KV,  #
                     HEAD_DIM: tl.constexpr,  #
                     BLOCK_M: tl.constexpr,  #
                     BLOCK_N: tl.constexpr,  #
                     GROUP_TILES: tl.constexpr,  #
                     should_mask_q: tl.constexpr,  #
                     ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h = off_hz % H_Q
    # GQA mapping: each KV head serves H_Q / H_KV consecutive Q heads.
    off_h_kv = off_h // (H_Q // H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h_kv.to(tl.int64) * stride_kh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m < N_CTX_Q)[:, None], other=0.0)

    # combined exp2-domain logsumexp per query row; padded rows load a huge
    # value so their probs underflow to exactly 0
    lse_offset = off_z.to(tl.int64) * lse_stride_b + off_h.to(tl.int64) * lse_stride_h + offs_m * lse_stride_n
    lse2 = tl.load(Lse2 + lse_offset, mask=(offs_m < N_CTX_Q), other=1.0e6)

    # blocksums base: GROUP_TILES consecutive 64-row programs share one query group.
    bsp_base = (blocksums_ptrs
                + off_z.to(tl.int64) * blocksums_stride_b
                + off_h.to(tl.int64) * blocksums_stride_h
                + (start_m // GROUP_TILES).to(tl.int64) * blocksums_stride_m)

    # full KV blocks (unmasked) then the trailing partial block (masked)
    _colsum_only_inner(q, lse2,  #
                       bsp_base, blocksums_stride_n,  #
                       K, k_offset, stride_kn, stride_kk,  #
                       qk_scale,  #
                       BLOCK_M, HEAD_DIM, BLOCK_N,  #
                       3, offs_m, offs_n, offs_d, N_CTX_Q, N_CTX_KV, False, should_mask_q)
    _colsum_only_inner(q, lse2,  #
                       bsp_base, blocksums_stride_n,  #
                       K, k_offset, stride_kn, stride_kk,  #
                       qk_scale,  #
                       BLOCK_M, HEAD_DIM, BLOCK_N,  #
                       4, offs_m, offs_n, offs_d, N_CTX_Q, N_CTX_KV, True, should_mask_q)


def colsum_only_attn(q, k, lse2, scale=None, group_rows=192):
    """Exact per-query-group softmax column-sums via a single QK^T sweep.

    :param q: [Z, H_Q, N_CTX_Q, D]
    :param k: [Z, H_KV, N_CTX_KV, D]; GQA-native (H_Q % H_KV == 0)
    :param lse2: [Z, H_Q, N_CTX_Q] fp32 — combined exp2-domain row logsumexp
        (natural-log lse * log2(e); equals M + log2(L) of attn_dense.py)
    :return: blocksums [Z, H_Q, ceil(N_CTX_Q / group_rows), N_CTX_KV] fp32
    """
    sm_scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    HEAD_DIM = q.shape[-1]
    assert q.ndim == 4 and k.ndim == 4
    assert q.shape[0] == k.shape[0]
    assert q.shape[-1] == k.shape[-1]
    assert q.shape[1] >= k.shape[1] and q.shape[1] % k.shape[1] == 0
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    assert lse2.shape == (q.shape[0], q.shape[1], q.shape[2]), \
        f'lse2 shape mismatch - lse2: {tuple(lse2.shape)}, q: {tuple(q.shape)}'
    assert lse2.dtype == torch.float32, f'lse2 must be fp32, got {lse2.dtype}'
    should_mask_q = q.shape[-2] % 64 != 0
    # query-group rows for the colsum output; must be a multiple of the pinned
    # BLOCK_M=64 tile (GROUP_TILES programs share one blocksums row via atomic_add)
    group_rows = int(group_rows)
    assert group_rows > 0 and group_rows % 64 == 0, \
        f'colsum group_rows must be a positive multiple of 64, got {group_rows}'
    mb = triton.cdiv(q.shape[2], group_rows)

    blocksums = torch.zeros((q.shape[0], q.shape[1], mb, k.shape[2]), device=q.device, dtype=torch.float32)

    grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    _colsum_only_fwd[grid](
        q, k, sm_scale, lse2,  #
        blocksums,  #
        lse2.stride(0), lse2.stride(1), lse2.stride(2),  #
        blocksums.stride(0), blocksums.stride(1), blocksums.stride(2), blocksums.stride(3),  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        q.shape[0], q.shape[1], k.shape[1],  #
        N_CTX_Q=q.shape[2],  #
        N_CTX_KV=k.shape[2],  #
        HEAD_DIM=HEAD_DIM,  #
        GROUP_TILES=group_rows // 64,
        should_mask_q=should_mask_q,
    )

    return blocksums


def main():
    """
    Smoke test on a GQA shape with q_len % 64 != 0, against an fp32 reference.
    """
    import torch.nn.functional as F

    torch.set_default_device('cuda')
    torch.manual_seed(0)

    B, HQ, HKV, NQ, NKV, D = 1, 8, 2, 1234, 4096, 128
    group_rows = 192
    q = torch.randn(B, HQ, NQ, D, dtype=torch.bfloat16)
    k = torch.randn(B, HKV, NKV, D, dtype=torch.bfloat16)
    scale = D ** -0.5

    k_exp = k.repeat_interleave(HQ // HKV, dim=1)
    s = torch.einsum('bhnd,bhmd->bhnm', q.float(), k_exp.float()) * scale
    lse2 = (torch.logsumexp(s, dim=-1) * 1.4426950408889634).float().contiguous()

    cs = colsum_only_attn(q, k, lse2, scale=scale, group_rows=group_rows)

    p = torch.softmax(s, dim=-1)
    mb = cdiv(NQ, group_rows)
    p_pad = F.pad(p, (0, 0, 0, mb * group_rows - NQ))
    cs_ref = p_pad.view(B, HQ, mb, group_rows, NKV).sum(3)

    err = (cs - cs_ref).abs().max().item()
    rel = (cs - cs_ref).abs().sum().item() / cs_ref.abs().sum().item()
    print(f'{cs.shape=} max_abs_err={err:.3e} rel_l1_err={rel:.3e}')
    assert rel < 1e-2, "colsum_only_attn not close to fp32 reference"
    print('colsum_only_attn smoke test PASSED')


if __name__ == '__main__':
    main()
