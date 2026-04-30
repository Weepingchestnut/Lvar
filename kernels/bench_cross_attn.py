import math
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import chipmunk
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Import flash_attn's attention
# from flash_attn_interface import flash_attn_func; print('#'*10 + ' use FA3 ' + '#'*10)
from flash_attn import flash_attn_func  # q, k, or v: BLHc, ret: BLHc

from kernels.sparse_attn.sparse_attn_ops import dense_attn
from kernels.flash_atten.triton.fused_attention import flash_attn_triton


# =========================
# Config
# =========================
@dataclass
class BenchConfig:
    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int
    # dtype: torch.dtype = torch.float16
    dtype: torch.dtype = torch.bfloat16


# 你可以按需改这里
BENCH_CONFIGS: List[BenchConfig] = [
    # NOTE: chipmunk do not support head_dim != 128
    BenchConfig(batch=1, heads=8,  q_len=256,  kv_len=256,  head_dim=64),
    BenchConfig(batch=1, heads=8,  q_len=256,  kv_len=1024, head_dim=64),
    BenchConfig(batch=2, heads=16, q_len=512,  kv_len=512,  head_dim=64),
    BenchConfig(batch=2, heads=16, q_len=512,  kv_len=2048, head_dim=64),
    
    BenchConfig(batch=4, heads=16, q_len=1024, kv_len=1024, head_dim=128),
    BenchConfig(batch=4, heads=16, q_len=1024, kv_len=4096, head_dim=128),
    
    BenchConfig(batch=1, heads=24,  q_len=2048,  kv_len=6425, head_dim=128),
    BenchConfig(batch=2, heads=24,  q_len=2048,  kv_len=6425, head_dim=128),
    BenchConfig(batch=1, heads=24,  q_len=4096,  kv_len=10521, head_dim=128),
    BenchConfig(batch=2, heads=24,  q_len=4096,  kv_len=10521, head_dim=128),

    # BenchConfig(batch=2, heads=32,  q_len=11520,  kv_len=15244, head_dim=128),
    # BenchConfig(batch=2, heads=32,  q_len=20640,  kv_len=24364, head_dim=128),
    # BenchConfig(batch=2, heads=32,  q_len=31800,  kv_len=35524, head_dim=128),
    # BenchConfig(batch=2, heads=32,  q_len=72000,  kv_len=75724, head_dim=128),

    # test Triton fused-attention
    # BenchConfig(batch=1, heads=24,  q_len=2048,  kv_len=2048, head_dim=128),
    # BenchConfig(batch=2, heads=24,  q_len=2048,  kv_len=2048, head_dim=128),
    # BenchConfig(batch=1, heads=24,  q_len=4096,  kv_len=4096, head_dim=128),
    # BenchConfig(batch=2, heads=24,  q_len=4096,  kv_len=4096, head_dim=128),
    # BenchConfig(batch=1, heads=24,  q_len=8192,  kv_len=8192, head_dim=128),
    # BenchConfig(batch=2, heads=24,  q_len=8192,  kv_len=8192, head_dim=128),
]

DEVICE = "cuda"
WARMUP = 20
ITERS = 100
CHECK_ACCURACY = True
ACCURACY_ATOL_FP16 = 2e-2
ACCURACY_RTOL_FP16 = 2e-2
PRINT_TENSOR_STATS = False


# =========================
# Utilities
# =========================
def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")


def supports_flash_triton_cfg(cfg: BenchConfig) -> bool:
    # 官方 flash_attn_triton 注释里强调 headdim <= 128 更稳妥
    # 这里保守一点
    return cfg.head_dim <= 128


def make_inputs(cfg: BenchConfig, requires_grad: bool = False):
    shape_q = (cfg.batch, cfg.heads, cfg.q_len, cfg.head_dim)
    shape_kv = (cfg.batch, cfg.heads, cfg.kv_len, cfg.head_dim)

    q = torch.randn(shape_q, device=DEVICE, dtype=cfg.dtype, requires_grad=requires_grad)
    k = torch.randn(shape_kv, device=DEVICE, dtype=cfg.dtype, requires_grad=requires_grad)
    v = torch.randn(shape_kv, device=DEVICE, dtype=cfg.dtype, requires_grad=requires_grad)
    return q, k, v


def flops_forward(cfg: BenchConfig) -> float:
    # 近似 FLOPs:
    # QK^T: 2 * B * H * Q * KV * D
    # P @ V: 2 * B * H * Q * KV * D
    # 合计: 4 * B * H * Q * KV * D
    return 4.0 * cfg.batch * cfg.heads * cfg.q_len * cfg.kv_len * cfg.head_dim


def tflops_from_ms(cfg: BenchConfig, ms: float) -> float:
    return flops_forward(cfg) / (ms * 1e-3) / 1e12


def benchmark_forward(fn: Callable[[], torch.Tensor], warmup: int = WARMUP, iters: int = ITERS) -> Tuple[float, float]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        del out
    torch.cuda.synchronize()

    # benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()

    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters
    return avg_ms, total_ms


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def rel_l2_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.norm((a - b).float()).item()
    den = torch.linalg.norm(b.float()).item()
    return num / max(den, eps)


# =========================
# Reference
# =========================
def ref_attention_fp32(q_bhqd: torch.Tensor, k_bhkd: torch.Tensor, v_bhkd: torch.Tensor, sm_scale: Optional[float]) -> torch.Tensor:
    """
    q: [B, H, Q, D]
    k: [B, H, KV, D]
    v: [B, H, KV, D]
    return: [B, H, Q, D]
    """
    q = q_bhqd.float()
    k = k_bhkd.float()
    v = v_bhkd.float()

    scale = sm_scale if sm_scale is not None else (1.0 / math.sqrt(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out


# =========================
# Providers
# =========================
def run_sdpa(q_bhqd: torch.Tensor, k_bhkd: torch.Tensor, v_bhkd: torch.Tensor, sm_scale: Optional[float]) -> torch.Tensor:
    # PyTorch SDPA expects [B, H, L, D]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q_bhqd, k_bhkd, v_bhkd,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=sm_scale,
        )


def run_flash_cuda(q_bhqd: torch.Tensor, k_bhkd: torch.Tensor, v_bhkd: torch.Tensor, sm_scale: Optional[float]) -> torch.Tensor:
    # flash-attn expects [B, L, H, D]
    q = q_bhqd.transpose(1, 2).contiguous()
    k = k_bhkd.transpose(1, 2).contiguous()
    v = v_bhkd.transpose(1, 2).contiguous()

    out = flash_attn_func(
        q, k, v,
        softmax_scale=sm_scale,
        causal=False,
    )
    return out.transpose(1, 2).contiguous()


def run_flash_triton(q_bhqd: torch.Tensor, k_bhkd: torch.Tensor, v_bhkd: torch.Tensor, sm_scale: Optional[float]) -> torch.Tensor:
    out, _ = dense_attn(q_bhqd, k_bhkd, v_bhkd, scale=sm_scale)

    return out


# --- Triton fused-attention ---
# def run_flash_triton(q_bhqd: torch.Tensor, k_bhkd: torch.Tensor, v_bhkd: torch.Tensor, sm_scale: Optional[float]) -> torch.Tensor:
#     out = flash_attn_triton(q_bhqd, k_bhkd, v_bhkd, scale=sm_scale)

#     return out


def run_chipmunk(q_bhqd: torch.Tensor, k_bhkd: torch.Tensor, v_bhkd: torch.Tensor, sm_scale: Optional[float]) -> torch.Tensor:
    # sm_scale == 1.0
    # out, _ = torch.ops.chipmunk.dense_attn(q_bhqd, k_bhkd, v_bhkd)

    # 20260425 new version support custom sm_scale
    out, _ = torch.ops.chipmunk.dense_attn(q_bhqd, k_bhkd, v_bhkd, scale=sm_scale)

    return out


PROVIDERS: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[float]], torch.Tensor]] = {
    "sdpa": run_sdpa,
    "flash_cuda": run_flash_cuda,
    # "flash_triton": run_flash_triton,
    "chipmunk": run_chipmunk,
}


# =========================
# Main evaluation
# =========================
def check_accuracy_one(cfg: BenchConfig, sm_scale: Optional[float] = None):
    q, k, v = make_inputs(cfg, requires_grad=False)

    ref = ref_attention_fp32(q, k, v, sm_scale).to(cfg.dtype)

    results = {}
    for name, fn in PROVIDERS.items():

        try:
            with torch.inference_mode():
                out = fn(q, k, v, sm_scale)
            torch.cuda.synchronize()

            ok = torch.allclose(
                out, ref,
                atol=ACCURACY_ATOL_FP16 if cfg.dtype == torch.bfloat16 or cfg.dtype == torch.float16 else 1e-4,
                rtol=ACCURACY_RTOL_FP16 if cfg.dtype == torch.bfloat16 or cfg.dtype == torch.float16 else 1e-4,
            )
            results[name] = {
                "ok": ok,
                "max_abs": max_abs_diff(out, ref),
                "rel_l2": rel_l2_error(out, ref),
            }

            if PRINT_TENSOR_STATS:
                print(
                    f"[ACC-DEBUG] {name}: "
                    f"mean={out.float().mean().item():.6f}, "
                    f"std={out.float().std().item():.6f}"
                )
        except Exception as e:
            results[name] = {
                "ok": False,
                "error": repr(e),
            }

    return results


def benchmark_one(cfg: BenchConfig, sm_scale: Optional[float] = None):
    # q, k, v = make_inputs(cfg, requires_grad=False)

    rows = []
    for name, fn in PROVIDERS.items():
        # print(f"{name=}, {fn=}")
        q, k, v = make_inputs(cfg, requires_grad=False)
        try:
            avg_ms, _ = benchmark_forward(lambda: fn(q, k, v, sm_scale))
            tflops = tflops_from_ms(cfg, avg_ms)
            rows.append((name, f"{avg_ms:.3f}", f"{tflops:.2f}", "ok"))
        except Exception as e:
            rows.append((name, "ERR", "ERR", str(e).split("\n")[0]))

    return rows


def print_header():
    print("=" * 120)
    print("Cross-Attention Forward Benchmark")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__}")
    print("=" * 120)


def print_accuracy(cfg: BenchConfig, acc_results: Dict[str, dict]):
    print("\n" + "-" * 120)
    print(
        f"[Accuracy] "
        f"B={cfg.batch:>2} H={cfg.heads:>3} Q={cfg.q_len:>5} KV={cfg.kv_len:>5} D={cfg.head_dim:>4} dtype={str(cfg.dtype).replace('torch.', '')}"
    )
    print(f"{'provider':<15} {'ok':<8} {'max_abs':<14} {'rel_l2':<14} {'note'}")
    # for name in ["sdpa", "flash_cuda", "flash_triton", "chipmunk"]:
    for name, item in acc_results.items():
        # if name not in acc_results:
        #     continue
        # item = acc_results[name]
        if "error" in item:
            print(f"{name:<15} {'False':<8} {'-':<14} {'-':<14} {item['error']}")
        else:
            print(
                f"{name:<15} "
                f"{str(item['ok']):<8} "
                f"{item['max_abs']:<14.6e} "
                f"{item['rel_l2']:<14.6e} "
            )
    print("-" * 120)


def print_perf(cfg: BenchConfig, rows: List[Tuple[str, str, str, str]]):
    print(
        f"[Forward Perf] "
        f"B={cfg.batch:>2} H={cfg.heads:>3} Q={cfg.q_len:>5} KV={cfg.kv_len:>5} D={cfg.head_dim:>4} dtype={str(cfg.dtype).replace('torch.', '')}"
    )
    print(f"{'provider':<15} {'avg_ms':<12} {'TFLOPS':<12} {'note'}")
    for name, ms, tflops, note in rows:
        print(f"{name:<15} {ms:<12} {tflops:<12} {note}")
    print("-" * 120)


def main():
    ensure_cuda()
    set_seed(0)
    torch.set_grad_enabled(False)

    # 可选：提升 matmul 精度/性能表现一致性
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    print_header()

    for cfg in BENCH_CONFIGS:
        sm_scale = 1.0 / math.sqrt(cfg.head_dim)
        # sm_scale = 1.0

        if CHECK_ACCURACY:
            acc_results = check_accuracy_one(cfg, sm_scale=sm_scale)
            print_accuracy(cfg, acc_results)

        perf_rows = benchmark_one(cfg, sm_scale=sm_scale)
        print_perf(cfg, perf_rows)


if __name__ == "__main__":
    main()
