import argparse

import chipmunk  # noqa: F401 - importing registers torch.ops.chipmunk
import torch
import triton


BM = 192
CSP_KV_TILE = 112


def make_tensor(shape, noncontiguous):
    if not noncontiguous:
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    base = torch.randn((shape[2], shape[0], shape[1], shape[3]), device="cuda", dtype=torch.bfloat16)
    return base.permute(1, 2, 0, 3)


def repeat_kv_for_q_heads(x, hq):
    hr = hq // x.shape[1]
    return x.repeat_interleave(hr, dim=1)


def dense_ref(q, k, v, scale):
    k = repeat_kv_for_q_heads(k, q.shape[1])
    v = repeat_kv_for_q_heads(v, q.shape[1])
    scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def lse_recip_ref(q, k, scale):
    k = repeat_kv_for_q_heads(k, q.shape[1])
    scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
    return torch.exp(scores).sum(dim=-1, keepdim=True).reciprocal()


def colsum_ref(q, k, prev_lse_recip, scale):
    b, hq, nq, _ = q.shape
    nkv = k.shape[2]
    qg = triton.cdiv(nq, BM)
    k = repeat_kv_for_q_heads(k, hq)
    scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
    weighted = torch.exp(scores) * prev_lse_recip.float()
    if qg * BM != nq:
        pad = qg * BM - nq
        weighted = torch.nn.functional.pad(weighted, (0, 0, 0, pad))
    return weighted.view(b, hq, qg, BM, nkv).sum(dim=3).to(torch.bfloat16)


def build_topk_indices(cs, sparse_cols):
    indices = cs.float().topk(k=sparse_cols, dim=-1).indices.to(torch.int32).contiguous()
    counts = torch.full(indices.shape[:-1], sparse_cols, device=indices.device, dtype=torch.int32)
    return indices, counts


def sparse_ref(q, k, v, indices, scale):
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
        idx = indices[:, :, group, :].long()
        gather_idx = idx[..., None].expand(-1, -1, -1, d)
        k_sel = torch.gather(k, 2, gather_idx)
        v_sel = torch.gather(v, 2, gather_idx)
        scores = torch.einsum("bhqd,bhkd->bhqk", q[:, :, q_start:q_end, :].float(), k_sel.float()) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, :, q_start:q_end, :] = torch.einsum("bhqk,bhkd->bhqd", probs, v_sel.float())
    return out.to(q.dtype)


def report(name, actual, expected):
    diff = (actual.float() - expected.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"{name:22s} max={max_diff:.6f} mean={mean_diff:.6f}")
    return max_diff, mean_diff


def assert_close(name, actual, expected, max_tol, mean_tol):
    max_diff, mean_diff = report(name, actual, expected)
    assert max_diff <= max_tol and mean_diff <= mean_tol, (
        f"{name} exceeded tolerance: max={max_diff:.6f}, mean={mean_diff:.6f}"
    )


def bench(name, fn, warmup, rep):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    print(f"{name:22s} {ms:.4f} ms")
    return ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hq", type=int, default=8)
    parser.add_argument("--hkv", type=int, default=4)
    parser.add_argument("--nq", type=int, default=577)
    parser.add_argument("--nkv", type=int, default=777)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--scale", type=float, default=0.0625)
    parser.add_argument("--sparse-cols", type=int, default=224)
    parser.add_argument("--noncontiguous", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    assert args.dim == 128, "CUDA kernels currently require head_dim == 128"
    assert args.hq % args.hkv == 0, "Hq must be divisible by Hkv"
    assert args.sparse_cols % CSP_KV_TILE == 0, "sparse-cols must be a multiple of the CSP KV tile"
    assert 0 < args.sparse_cols <= args.nkv, "sparse-cols must be in (0, Nkv]"

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    shape_q = (args.batch, args.hq, args.nq, args.dim)
    shape_kv = (args.batch, args.hkv, args.nkv, args.dim)
    q0 = make_tensor(shape_q, args.noncontiguous)
    k0 = make_tensor(shape_kv, args.noncontiguous)
    v0 = make_tensor(shape_kv, args.noncontiguous)
    q1 = make_tensor(shape_q, args.noncontiguous)
    k1 = make_tensor(shape_kv, args.noncontiguous)
    v1 = make_tensor(shape_kv, args.noncontiguous)

    with torch.no_grad():
        o0, lse0 = torch.ops.chipmunk.dense_attn(q0, k0, v0, args.scale)
        o0_ref = dense_ref(q0, k0, v0, args.scale)
        lse0_ref = lse_recip_ref(q0, k0, args.scale)
        assert_close("dense_attn/o", o0, o0_ref, 0.30, 0.025)
        assert_close("dense_attn/lse", lse0, lse0_ref, 0.02, 0.002)

        o1, cs, lse1 = torch.ops.chipmunk.dense_colsum_attn(q1, k1, v1, lse0, args.scale)
        o1_ref = dense_ref(q1, k1, v1, args.scale)
        lse1_ref = lse_recip_ref(q1, k1, args.scale)
        cs_ref = colsum_ref(q1, k1, lse0, args.scale)
        assert cs.shape == (args.batch, args.hq, triton.cdiv(args.nq, BM), args.nkv)
        assert_close("dense_colsum/o", o1, o1_ref, 0.30, 0.025)
        assert_close("dense_colsum/lse", lse1, lse1_ref, 0.02, 0.002)
        assert_close("dense_colsum/cs", cs, cs_ref, 0.35, 0.035)

        indices, counts = build_topk_indices(cs, args.sparse_cols)
        sparse = torch.zeros_like(q1)
        torch.ops.chipmunk.csp_attn(q1, k1, v1, sparse, indices, counts, 1, args.scale)
        sparse_expected = sparse_ref(q1, k1, v1, indices, args.scale)
        assert_close("csp_attn/sparse", sparse, sparse_expected, 0.35, 0.035)

        cache = o1.clone()
        torch.ops.chipmunk.csp_attn(q1, k1, v1, cache, indices, counts, -1, args.scale)
        torch.ops.chipmunk.csp_attn(q1, k1, v1, cache, indices, counts, 1, args.scale)
        assert_close("cache roundtrip", cache, o1, 0.05, 0.005)

        if not args.skip_bench:
            print("\nLatency")
            k1_rep = repeat_kv_for_q_heads(k1, args.hq)
            v1_rep = repeat_kv_for_q_heads(v1, args.hq)
            csp_bench_out = torch.empty_like(q1)

            def run_csp_add():
                csp_bench_out.zero_()
                torch.ops.chipmunk.csp_attn(q1, k1, v1, csp_bench_out, indices, counts, 1, args.scale)

            bench("torch sdpa", lambda: torch.nn.functional.scaled_dot_product_attention(q1, k1_rep, v1_rep, scale=args.scale), args.warmup, args.rep)
            bench("dense_attn", lambda: torch.ops.chipmunk.dense_attn(q1, k1, v1, args.scale), args.warmup, args.rep)
            bench("dense_colsum_attn", lambda: torch.ops.chipmunk.dense_colsum_attn(q1, k1, v1, lse0, args.scale), args.warmup, args.rep)
            bench("csp_attn add", run_csp_add, args.warmup, args.rep)


if __name__ == "__main__":
    main()
