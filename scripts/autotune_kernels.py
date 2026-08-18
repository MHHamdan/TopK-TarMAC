"""Search the same tiling space for the dense and the sparse kernel, and report
what fraction of the machine each one reaches.

A hand-written sparse kernel compared against a hand-written dense kernel is
only a fair comparison if both have had the same opportunity to be fast. Matched
authorship is not enough: the identifying condition is matched efficiency
against what the machine demonstrably attains on the same shape. This script
establishes that condition, or fails to, by giving both kernels the same search
budget over the same parameters and reporting achieved arithmetic rate beside
the latency.

The reference point is what the device actually reaches on this problem, taken
as the best over every implementation available -- including the vendor library
at the head widths where it is strong -- rather than a datasheet number.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kernels import fused_dense_attention as fda  # noqa: E402
from src.kernels import fused_topk_attention as fta  # noqa: E402


def _time(fn, device: str, reps: int = 40, warmup: int = 15) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(reps):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(device)
        ts.append(s.elapsed_time(e))
    return min(ts)


def peak_flops(device: str) -> float:
    n = 8192
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    t = _time(lambda: a @ b, device, reps=20, warmup=5)
    del a, b
    torch.cuda.empty_cache()
    return 2 * n ** 3 / (t * 1e-3)


def dense_variants(q, k, v, device):
    """Every dense implementation we can reach, with its tiling parameters."""
    B, N, d = q.shape
    out = {}
    mask = torch.zeros(N, N, device=device, dtype=q.dtype)
    mask.masked_fill_(torch.eye(N, device=device, dtype=torch.bool),
                      float("-inf"))
    out["sdpa"] = (lambda: F.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
        attn_mask=mask).squeeze(1), {})
    bd = max(16, fda._next_pow2(d))
    for bm, bkn, warps, stages in itertools.product(
            (16, 32, 64, 128), (32, 64, 128), (2, 4, 8), (1, 2, 3)):
        if bm > fda._next_pow2(N) or bkn > fda._next_pow2(N):
            continue
        if (2 * bkn * bd + bm * bd) * 4 * stages > 100_000:
            continue
        cfg = {"BLOCK_M": bm, "BLOCK_KN": bkn, "num_warps": warps,
               "num_stages": stages}
        out[f"triton_m{bm}_n{bkn}_w{warps}_s{stages}"] = (
            lambda c=cfg: _dense_call(q, k, v, c), cfg)
    return out


def _dense_call(q, k, v, cfg):
    B, N, d = q.shape
    o = torch.empty_like(q)
    bd = max(16, fda._next_pow2(d))
    fda._flash_dense_kernel[(triton.cdiv(N, cfg["BLOCK_M"]), B)](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        N, 1.0 / (d ** 0.5),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_KN=cfg["BLOCK_KN"], BLOCK_D=bd,
        HEAD_D=d, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return o


def _sparse_call(q, k, v, topk, cfg):
    B, N, d = q.shape
    o = torch.empty_like(q)
    bn, bd = fta._next_pow2(N), max(16, fta._next_pow2(d))
    fta._fused_topk_tiled_kernel[(triton.cdiv(N, cfg["BLOCK_M"]), B)](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        N, 1.0 / (d ** 0.5),
        TOPK=topk, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=bn, BLOCK_D=bd,
        BLOCK_DC=cfg["BLOCK_DC"], HEAD_D=d,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return o


def sparse_variants(q, k, v, topk, device):
    B, N, d = q.shape
    bn, bd = fta._next_pow2(N), max(16, fta._next_pow2(d))
    out = {}
    for bm, bdc, warps, stages in itertools.product(
            (16, 32, 64, 128), (16, 32, 64), (2, 4, 8), (1, 2, 3)):
        if bm > bn or bdc > bd:
            continue
        if (bm * bn + bn * bdc + bm * bd) * 4 * stages > 100_000:
            continue
        cfg = {"BLOCK_M": bm, "BLOCK_DC": bdc, "num_warps": warps,
               "num_stages": stages}
        out[f"triton_m{bm}_dc{bdc}_w{warps}_s{stages}"] = (
            lambda c=cfg: _sparse_call(q, k, v, topk, c), cfg)
    return out


def search(variants, ref, device, reps) -> list[dict[str, Any]]:
    results = []
    for name, (fn, cfg) in variants.items():
        try:
            got = fn()
            err = (got - ref).abs().max().item()
            if err > 1e-4:
                results.append({"name": name, "cfg": cfg, "ms": None,
                                "error": f"deviates by {err:.2e}"})
                continue
            results.append({"name": name, "cfg": cfg,
                            "ms": _time(fn, device, reps), "deviation": err})
        except Exception as exc:
            results.append({"name": name, "cfg": cfg, "ms": None,
                            "error": f"{type(exc).__name__}: {exc}"[:120]})
        torch.cuda.empty_cache()
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-list", nargs="+", type=int, default=[192, 384, 512])
    p.add_argument("--k-list", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--d-list", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--out", default=str(ROOT / "results" / "autotune.json"))
    args = p.parse_args()

    device = args.device
    torch.cuda.set_device(device)
    peak = peak_flops(device)
    print(f"measured peak fp32 matrix-multiply rate: {peak / 1e12:.1f} TFLOP/s\n")

    rows = []
    t0 = time.time()
    for d in args.d_list:
        for N in args.n_list:
            B = args.batch
            g = torch.Generator(device=device).manual_seed(7)
            q = torch.randn(B, N, d, device=device, generator=g)
            k = torch.randn(B, N, d, device=device, generator=g)
            v = torch.randn(B, N, d, device=device, generator=g)
            eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
            s = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
            s = s.masked_fill(eye, float("-inf"))
            ref_dense = torch.matmul(F.softmax(s, dim=-1), v)

            dv = search(dense_variants(q, k, v, device), ref_dense, device,
                        args.reps)
            ok = [r for r in dv if r["ms"]]
            best_d = min(ok, key=lambda r: r["ms"])
            dense_flops = 4 * B * N * N * d
            print(f"d={d:>3} N={N:>4}  dense best {best_d['ms']:.4f} ms "
                  f"({best_d['name']}) = "
                  f"{dense_flops / (best_d['ms'] * 1e-3) / 1e12:5.1f} TFLOP/s, "
                  f"{100 * dense_flops / (best_d['ms'] * 1e-3) / peak:4.1f}% of peak "
                  f"[{len(ok)}/{len(dv)} configs ran]", flush=True)
            rows.append({"kind": "dense", "head_dim": d, "n_agents": N,
                         "batch": B, "peak_flops": peak,
                         "best": best_d, "n_configs": len(dv),
                         "n_ran": len(ok),
                         "all": sorted(ok, key=lambda r: r["ms"])[:5]})

            for topk in args.k_list:
                if topk >= N:
                    continue
                ref_sp = fta.reference_topk_attention(q, k, v, topk)
                sv = search(sparse_variants(q, k, v, topk, device), ref_sp,
                            device, args.reps)
                oks = [r for r in sv if r["ms"]]
                if not oks:
                    continue
                best_s = min(oks, key=lambda r: r["ms"])
                sp_flops = 2 * B * N * N * d + 2 * B * N * topk * d
                print(f"          k={topk:<4} sparse best {best_s['ms']:.4f} ms "
                      f"({best_s['name']}) = "
                      f"{sp_flops / (best_s['ms'] * 1e-3) / 1e12:5.1f} TFLOP/s, "
                      f"{100 * sp_flops / (best_s['ms'] * 1e-3) / peak:4.1f}% "
                      f"of peak | sparse/dense = "
                      f"{best_s['ms'] / best_d['ms']:.3f}", flush=True)
                rows.append({"kind": "sparse", "head_dim": d, "n_agents": N,
                             "batch": B, "k": topk, "peak_flops": peak,
                             "best": best_s, "best_dense_ms": best_d["ms"],
                             "n_configs": len(sv), "n_ran": len(oks),
                             "all": sorted(oks, key=lambda r: r["ms"])[:5]})
            del q, k, v, eye, s, ref_dense
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(
        {"device": device, "device_name": torch.cuda.get_device_name(device),
         "peak_flops": peak, "batch": args.batch, "reps": args.reps,
         "torch": torch.__version__, "triton": triton.__version__,
         "elapsed_s": round(time.time() - t0, 1), "rows": rows},
        indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
