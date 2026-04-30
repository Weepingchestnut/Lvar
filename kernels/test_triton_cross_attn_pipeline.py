import argparse
import math

import torch
import triton

from kernels.sparse_attn.triton import csp_attn
from kernels.sparse_attn.triton import dense_attn
from kernels.sparse_attn.triton import dense_colsum_attn


BM = 192


def make_tensor(shape, noncontiguous):
    if not noncontiguous:
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    base = torch.randn((shape[2], shape[0], shape[1], shape[3]), device="cuda", dtype=torch.bfloat16)
    return base.permute(1, 2, 0, 3)


def repeat_kv_for_q_heads(x, hq):
    assert hq % x.shape[1] == 0
    return x.repeat_interleave(hq // x.shape[1], dim=1)


def scores_log2(q, k, scale):
    k = repeat_kv_for_q_heads(k, q.shape[1])
    return torch.matmul(q.float(), k.transpose(-2, -1).float()) * (scale * math.log2(math.e))


def dense_ref(q, k, v, scale):
    k = repeat_kv_for_q_heads(k, q.shape[1])
    v = repeat_kv_for_q_heads(v, q.shape[1])
    scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def lse_ref(q, k, scale):
    s = scores_log2(q, k, scale)
    m = s.max(dim=-1, keepdim=True).values
    l = torch.exp2(s - m).sum(dim=-1, keepdim=True)
    return m, l


def colsum_ref(q, k, prev_lse, scale):
    b, hq, nq, _ = q.shape
    nkv = k.shape[2]
    qg = triton.cdiv(nq, BM)
    prev_m, prev_l = prev_lse
    weighted = torch.exp2(scores_log2(q, k, scale) - prev_m.float()) / prev_l.float()
    if qg * BM != nq:
        weighted = torch.nn.functional.pad(weighted, (0, 0, 0, qg * BM - nq))
    return weighted.view(b, hq, qg, BM, nkv).sum(dim=3)


def sparse_ref(q, k, v, indices, counts, scale):
    b, hq, nq, d = q.shape
    qg = indices.shape[2]
    k = repeat_kv_for_q_heads(k, hq)
    v = repeat_kv_for_q_heads(v, hq)
    out = torch.empty((b, hq, nq, d), device=q.device, dtype=torch.float32)

    for group in range(qg):
        q_start = group * BM
        q_end = min(q_start + BM, nq)
        if q_start >= q_end:
            continue

        count = counts[:, :, group]
        assert torch.all(count == count.flatten()[0]), "This reference expects uniform sparse counts"
        sparse_cols = int(count.flatten()[0].item())
        idx = indices[:, :, group, :sparse_cols].long()
        gather_idx = idx[..., None].expand(-1, -1, -1, d)
        k_sel = torch.gather(k, 2, gather_idx)
        v_sel = torch.gather(v, 2, gather_idx)
        q_block = q[:, :, q_start:q_end, :]
        scores = torch.einsum("bhqd,bhkd->bhqk", q_block.float(), k_sel.float()) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, :, q_start:q_end, :] = torch.einsum("bhqk,bhkd->bhqd", probs, v_sel.float())

    return out.to(q.dtype)


def build_topk_indices(colsum, sparse_cols):
    indices = colsum.float().topk(k=sparse_cols, dim=-1).indices.to(torch.int32).contiguous()
    counts = torch.full(indices.shape[:-1], sparse_cols, device=indices.device, dtype=torch.int32)
    assert int(indices.min().item()) >= 0
    assert int(indices.max().item()) < colsum.shape[-1]
    return indices, counts


def report(name, actual, expected):
    diff = (actual.float() - expected.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"{name:24s} max={max_diff:.6f} mean={mean_diff:.6f}")
    return max_diff, mean_diff


def assert_close(name, actual, expected, max_tol, mean_tol):
    max_diff, mean_diff = report(name, actual, expected)
    assert max_diff <= max_tol and mean_diff <= mean_tol, (
        f"{name} exceeded tolerance: max={max_diff:.6f}, mean={mean_diff:.6f}"
    )


def bench(name, fn, warmup, rep):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    print(f"{name:24s} {ms:.4f} ms")
    return ms


def run_correctness(args):
    scale = 1.0 / math.sqrt(args.dim) if args.scale is None else args.scale
    shape_q = (args.batch, args.hq, args.nq, args.dim)
    shape_kv = (args.batch, args.hkv, args.nkv, args.dim)

    q = make_tensor(shape_q, args.noncontiguous)
    k = make_tensor(shape_kv, args.noncontiguous)
    v = make_tensor(shape_kv, args.noncontiguous)

    with torch.no_grad():
        o_dense, lse = dense_attn(q, k, v, scale)
        o_dense_ref = dense_ref(q, k, v, scale)
        m_ref, l_ref = lse_ref(q, k, scale)
        assert_close("dense/o", o_dense, o_dense_ref, args.out_max_tol, args.out_mean_tol)
        assert_close("dense/m", lse[0], m_ref, args.lse_max_tol, args.lse_mean_tol)
        assert_close("dense/l", lse[1], l_ref, args.lse_max_tol, args.lse_mean_tol)

        # Chipmunk's full step computes colsum from the same attention
        # probabilities whose LSE was produced by the dense pass.
        o_full, colsum, lse_full = dense_colsum_attn(q, k, v, lse, scale)
        expected_colsum_shape = (args.batch, args.hq, triton.cdiv(args.nq, BM), args.nkv)
        assert tuple(colsum.shape) == expected_colsum_shape, (
            f"colsum shape mismatch: got {tuple(colsum.shape)}, expected {expected_colsum_shape}"
        )

        colsum_expected = colsum_ref(q, k, lse, scale)
        assert_close("dense_colsum/o", o_full, o_dense_ref, args.out_max_tol, args.out_mean_tol)
        assert_close("dense_colsum/m", lse_full[0], m_ref, args.lse_max_tol, args.lse_mean_tol)
        assert_close("dense_colsum/l", lse_full[1], l_ref, args.lse_max_tol, args.lse_mean_tol)
        assert_close("dense_colsum/cs", colsum, colsum_expected, args.cs_max_tol, args.cs_mean_tol)

        indices, counts = build_topk_indices(colsum, args.sparse_cols)
        sparse, _ = csp_attn(q, k, v, indices, counts, 1.0, scale)
        sparse_expected = sparse_ref(q, k, v, indices, counts, scale)
        assert_close("csp/sparse", sparse, sparse_expected, args.out_max_tol, args.out_mean_tol)

        # This mirrors Chipmunk's delta-attention algebra for one frozen step:
        # O_cache = O_dense - O_sparse; O_cache + O_sparse must recover O_dense.
        cache = o_full.float() - sparse.float()
        restored = (cache + sparse.float()).to(o_full.dtype)
        assert_close("cache roundtrip", restored, o_full, 0.01, 0.001)

    return q, k, v, lse, indices, counts, scale


def run_benchmarks(args, tensors):
    q, k, v, prev_lse, indices, counts, scale = tensors
    k_rep = repeat_kv_for_q_heads(k, q.shape[1])
    v_rep = repeat_kv_for_q_heads(v, q.shape[1])

    print("\nLatency")
    sdpa_ms = bench(
        "torch sdpa",
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, scale=scale),
        args.warmup,
        args.rep,
    )
    dense_ms = bench("triton dense", lambda: dense_attn(q, k, v, scale), args.warmup, args.rep)
    colsum_ms = bench("triton dense+cs", lambda: dense_colsum_attn(q, k, v, prev_lse, scale), args.warmup, args.rep)
    sparse_ms = bench("triton csp sparse", lambda: csp_attn(q, k, v, indices, counts, 1.0, scale), args.warmup, args.rep)

    print("\nSpeedup")
    print(f"sdpa / dense        {sdpa_ms / dense_ms:.3f}x")
    print(f"dense / csp_sparse  {dense_ms / sparse_ms:.3f}x")
    print(f"sdpa / csp_sparse   {sdpa_ms / sparse_ms:.3f}x")
    print(f"dense+cs overhead   {colsum_ms / dense_ms:.3f}x dense")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--hq", type=int, default=32)
    parser.add_argument("--hkv", type=int, default=32)
    parser.add_argument("--nq", type=int, default=577)
    parser.add_argument("--nkv", type=int, default=777)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--sparse-cols", type=int, default=128)
    parser.add_argument("--noncontiguous", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-max-tol", type=float, default=0.35)
    parser.add_argument("--out-mean-tol", type=float, default=0.035)
    parser.add_argument("--lse-max-tol", type=float, default=0.06)
    parser.add_argument("--lse-mean-tol", type=float, default=0.006)
    parser.add_argument("--cs-max-tol", type=float, default=0.60)
    parser.add_argument("--cs-mean-tol", type=float, default=0.060)
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    assert args.dim in {16, 32, 64, 128, 256}
    assert args.hq % args.hkv == 0, "Hq must be divisible by Hkv for GQA mapping"
    assert 0 < args.sparse_cols <= args.nkv

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    effective_scale = 1.0 / math.sqrt(args.dim) if args.scale is None else args.scale

    print(
        "shape "
        f"B={args.batch} Hq={args.hq} Hkv={args.hkv} Nq={args.nq} Nkv={args.nkv} "
        f"D={args.dim} sparse_cols={args.sparse_cols} scale={effective_scale}"
    )
    tensors = run_correctness(args)
    if not args.skip_bench:
        run_benchmarks(args, tensors)


if __name__ == "__main__":
    main()
