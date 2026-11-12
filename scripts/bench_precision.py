"""Does the fused crossover survive a numerical format that gives dense attention
its tensor cores back?

Every latency in the main sweep is fp32 with ``input_precision="ieee"``, which
keeps both kernels off the tensor cores so that they reproduce the reference's
summation order. That is the format in which dense attention is least
advantaged, and the manuscript's positive result -- a fused top-k kernel faster
than the best dense kernel at one retained peer -- is only established there.
This script repeats the comparison in three formats:

    fp32   ieee products, as in the main sweep
    tf32   fp32 storage, TF32 products on the tensor cores, for every kernel
           (Triton ``input_precision="tf32"``; PyTorch's matmul TF32 switch on)
    bf16   bf16 storage and products, fp32 accumulation

TF32 is also the configuration that separates two readings the main sweep
cannot: it raises dense efficiency without changing the head width.

Both hand-written kernels are searched over the same tiling space under the
same shared-memory filter as ``autotune_kernels.py``, separately in every
format, and the winners are then timed interleaved with the vendor library,
FlexAttention and the composed top-k path. FlexAttention enters twice: as a
dense baseline with the diagonal masked, and as PyTorch's production route for
data-dependent sparsity, with the top-k retained set expressed as a block mask.

Outside fp32-ieee the top-k kernels need not select the same peers as the fp32
reference, because rounding can reorder near-tied scores. The deviation from a
float64 reference is recorded per implementation so that a latency is never
reported without the function it computed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kernels import fused_dense_attention as fda  # noqa: E402
from src.kernels import fused_topk_attention as fta  # noqa: E402

MODES = {"fp32": (torch.float32, "ieee", False),
         "tf32": (torch.float32, "tf32", True),
         "bf16": (torch.bfloat16, "ieee", False)}

# Every (N, d, dtype, mask) is a new specialisation. Past dynamo's default
# recompile limit flex_attention silently falls back to an unfused eager path
# that materialises the score matrix, which would time a different algorithm;
# raise the limit, and fail rather than fall back if it is ever reached.
torch._dynamo.config.recompile_limit = 100_000
torch._dynamo.config.accumulated_recompile_limit = 100_000
torch._dynamo.config.fail_on_recompile_limit_hit = True
_flex = torch.compile(flex_attention, dynamic=False)


def set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def time_min(fn: Callable, device: str, reps: int, warmup: int = 10) -> float:
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


def peak_flops(device: str, dtype: torch.dtype) -> float:
    n = 8192
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    t = time_min(lambda: a @ b, device, reps=20, warmup=5)
    del a, b
    torch.cuda.empty_cache()
    return 2 * n ** 3 / (t * 1e-3)


# ---------------------------------------------------------------------------
# kernels at an explicit tiling and precision
# ---------------------------------------------------------------------------

def dense_call(q, k, v, cfg, prec):
    B, N, d = q.shape
    o = torch.empty_like(q)
    bd = max(16, fda._next_pow2(d))
    fda._flash_dense_kernel[(triton.cdiv(N, cfg["BLOCK_M"]), B)](
        q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        N, 1.0 / (d ** 0.5),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_KN=cfg["BLOCK_KN"], BLOCK_D=bd,
        HEAD_D=d, PRECISION=prec, num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"])
    return o


def sparse_call(q, k, v, topk, cfg, prec):
    B, N, d = q.shape
    o = torch.empty_like(q)
    bn, bd = fta._next_pow2(N), max(16, fta._next_pow2(d))
    fta._fused_topk_tiled_kernel[(triton.cdiv(N, cfg["BLOCK_M"]), B)](
        q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        N, 1.0 / (d ** 0.5),
        TOPK=topk, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=bn, BLOCK_D=bd,
        BLOCK_DC=cfg["BLOCK_DC"], HEAD_D=d, PRECISION=prec,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return o


def dense_space(N: int, d: int):
    """The tiling space and admissibility filter of autotune_kernels.py."""
    bd = max(16, fda._next_pow2(d))
    for bm, bkn, w, st in itertools.product(
            (16, 32, 64, 128), (32, 64, 128), (2, 4, 8), (1, 2, 3)):
        if bm > fda._next_pow2(N) or bkn > fda._next_pow2(N):
            continue
        if (2 * bkn * bd + bm * bd) * 4 * st > 100_000:
            continue
        yield {"BLOCK_M": bm, "BLOCK_KN": bkn, "num_warps": w,
               "num_stages": st}


def sparse_space(N: int, d: int):
    bn, bd = fta._next_pow2(N), max(16, fta._next_pow2(d))
    for bm, bdc, w, st in itertools.product(
            (16, 32, 64, 128), (16, 32, 64), (2, 4, 8), (1, 2, 3)):
        if bm > bn or bdc > bd:
            continue
        if (bm * bn + bn * bdc + bm * bd) * 4 * st > 100_000:
            continue
        yield {"BLOCK_M": bm, "BLOCK_DC": bdc, "num_warps": w,
               "num_stages": st}


#: Largest fraction of query rows allowed to differ from the float64
#: reference by more than the format's tolerance. Exact top-k is discontinuous
#: in the scores: any rounding other than fp32-ieee reorders some near-tied
#: pairs and so retains a different peer in a few rows, which a maximum over
#: rows would report as a failure. A kernel that selected wrongly would differ
#: in most rows, not in a fraction of a per cent.
ROW_TOL = 1e-3


def disagreement(out, ref, tol):
    """(max deviation, fraction of rows deviating by more than tol)."""
    err = (out.float() - ref).abs().amax(-1)
    return err.max().item(), (err > tol).float().mean().item()


def search(space, call, ref, tol, device, reps):
    best, n_ok, n_all = None, 0, 0
    for cfg in space:
        n_all += 1
        try:
            dev, frac = disagreement(call(cfg), ref, tol)
            if not frac <= ROW_TOL:
                continue
            ms = time_min(lambda c=cfg: call(c), device, reps)
            n_ok += 1
            if best is None or ms < best["ms"]:
                best = {"cfg": cfg, "ms": ms, "deviation": dev,
                        "rows_differing": frac}
        except Exception:
            pass
        torch.cuda.empty_cache()
    return best, n_ok, n_all


# ---------------------------------------------------------------------------
# library, FlexAttention and composed paths
# ---------------------------------------------------------------------------

def sdpa(q, k, v, mask):
    return F.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
        attn_mask=mask).squeeze(1)


def flex_dense(q, k, v, bm):
    return _flex(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                 block_mask=bm).squeeze(1)


def rank_scores(q, k, eye):
    """Scores to rank, in fp32 whatever the storage format. A bf16 score matrix
    rounds near-tied scores together and then selects different peers from the
    reference; the fused kernel accumulates its scores in fp32, and the composed
    paths are given the same so that all of them compute one function."""
    d = q.shape[-1]
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) / d ** 0.5
    return s.masked_fill(eye, float("-inf"))


def flex_topk(q, k, v, topk, eye):
    """Top-k as PyTorch expresses data-dependent sparsity: rank the scores,
    build a block mask from the retained set, and let FlexAttention skip the
    blocks it can. The block mask depends on the data, so it is built inside
    the timed call."""
    B, N, d = q.shape
    idx = rank_scores(q, k, eye).topk(topk, dim=-1, sorted=False).indices
    keep = torch.zeros(B, N, N, device=q.device, dtype=torch.bool)
    keep.scatter_(-1, idx, True)

    def mm(b, h, qi, ki):
        return keep[b, qi, ki]

    bm = create_block_mask(mm, B, None, N, N, device=q.device, _compile=True)
    return _flex(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                 block_mask=bm).squeeze(1)


def topk_gather(q, k, v, topk, eye):
    B, N, d = q.shape
    ts, ti = rank_scores(q, k, eye).topk(topk, dim=-1, sorted=False)
    w = F.softmax(ts.float(), dim=-1).to(q.dtype)
    idx = ti.reshape(B, N * topk, 1).expand(B, N * topk, d)
    vg = v.gather(1, idx).view(B, N, topk, d)
    return torch.matmul(w.unsqueeze(-2), vg).squeeze(-2)


def references(q, k, v, ks):
    """float64 dense and top-k references from the same (rounded) inputs."""
    q, k, v = q.double(), k.double(), v.double()
    N, d = q.shape[1], q.shape[2]
    eye = torch.eye(N, device=q.device, dtype=torch.bool)
    s = (q @ k.transpose(-1, -2) / d ** 0.5).masked_fill(eye, float("-inf"))
    out = {"dense": (torch.softmax(s, -1) @ v).float()}
    for kk in ks:
        ts, ti = s.topk(kk, dim=-1)
        w = torch.softmax(ts, -1)
        vg = v.gather(1, ti.reshape(q.shape[0], -1, 1).expand(-1, -1, d)
                      ).view(q.shape[0], N, kk, d)
        out[kk] = (w.unsqueeze(-1) * vg).sum(-2).float()
    return out


# ---------------------------------------------------------------------------
# interleaved timing of the finalists
# ---------------------------------------------------------------------------

def interleaved(fns: dict[str, Callable], device: str, reps: int, warmup: int,
                runs: int) -> dict[str, dict[str, Any]]:
    def one(fn):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(device)
        return s.elapsed_time(e)

    for _ in range(warmup):
        for fn in fns.values():
            one(fn)
    blocks = {n: [] for n in fns}
    for _ in range(runs):
        cur = {n: [] for n in fns}
        for _ in range(reps):
            for n, fn in fns.items():
                cur[n].append(one(fn))
        for n in fns:
            blocks[n].append(cur[n])
    out = {}
    for n, bl in blocks.items():
        flat = sorted(x for b in bl for x in b)
        med = [statistics.median(b) for b in bl]
        out[n] = {"min_ms": flat[0], "median_ms": statistics.median(flat),
                  "run_medians_ms": med,
                  "inter_run_spread_pct":
                      100.0 * (max(med) - min(med)) / min(med)}
    return out


def dump(out, device, B, peaks, args, t0, rows) -> None:
    """Write what has been measured so far, so an interrupted sweep keeps it."""
    payload = {
        "device": device, "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": triton.__version__, "batch": B, "peaks": peaks,
        "reps": args.reps, "runs": args.runs,
        "timing": "cuda_events_interleaved_min",
        "elapsed_s": round(time.time() - t0, 1), "rows": rows,
    }
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--n-list", nargs="+", type=int, default=[192, 384, 512])
    p.add_argument("--k-list", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--d-list", nargs="+", type=int, default=[16, 32])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--search-reps", type=int, default=30)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "results" / "precision.json"))
    args = p.parse_args()

    device, B = args.device, args.batch
    torch.cuda.set_device(device)
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    peaks: dict[str, float] = {}

    for mode in args.modes:
        dtype, prec, tf32 = MODES[mode]
        set_tf32(tf32)
        peaks[mode] = peak_flops(device, dtype)
        # A tolerance that admits the format's own rounding and rejects a
        # kernel that has selected different peers from the reference.
        tol = {"fp32": 1e-4, "tf32": 2e-2, "bf16": 5e-2}[mode]
        for d in args.d_list:
            for N in args.n_list:
                gen = torch.Generator(device=device).manual_seed(77 + N + d)
                q, k, v = (torch.randn(B, N, d, device=device, generator=gen)
                           .to(dtype) for _ in range(3))
                eye = torch.eye(N, device=device, dtype=torch.bool)
                ks = [kk for kk in args.k_list if kk < N]
                refs = references(q, k, v, ks)
                mask = torch.zeros(N, N, device=device, dtype=dtype
                                   ).masked_fill_(eye, float("-inf"))
                bm_dense = create_block_mask(
                    lambda b, h, qi, ki: qi != ki, None, None, N, N,
                    device=device)

                dbest, dok, dall = search(
                    dense_space(N, d), lambda c: dense_call(q, k, v, c, prec),
                    refs["dense"], tol, device, args.search_reps)
                sbest = {}
                for kk in ks:
                    sbest[kk] = search(
                        sparse_space(N, d),
                        lambda c, kk=kk: sparse_call(q, k, v, kk, c, prec),
                        refs[kk], tol, device, args.search_reps)

                fns: dict[str, Callable] = {
                    "sdpa": lambda: sdpa(q, k, v, mask),
                    "flex_dense": lambda: flex_dense(q, k, v, bm_dense),
                }
                if dbest:
                    fns["dense_tuned"] = (
                        lambda c=dbest["cfg"]: dense_call(q, k, v, c, prec))
                for kk in ks:
                    if sbest[kk][0]:
                        fns[f"fused_topk:{kk}"] = (
                            lambda kk=kk, c=sbest[kk][0]["cfg"]:
                            sparse_call(q, k, v, kk, c, prec))
                    fns[f"topk_gather:{kk}"] = (
                        lambda kk=kk: topk_gather(q, k, v, kk, eye))
                    fns[f"flex_topk:{kk}"] = (
                        lambda kk=kk: flex_topk(q, k, v, kk, eye))

                deviation = {}
                with torch.no_grad():
                    for name, fn in list(fns.items()):
                        try:
                            ref = (refs["dense"] if ":" not in name
                                   else refs[int(name.split(":")[1])])
                            dmax, frac = disagreement(fn(), ref, tol)
                            deviation[name] = dmax
                            deviation[name + ":rows_differing"] = frac
                        except Exception as exc:
                            print(f"  {mode} d{d} N{N} {name} failed: "
                                  f"{type(exc).__name__}: {exc}"[:200])
                            fns.pop(name)
                    timed = interleaved(fns, device, args.reps, args.warmup,
                                        args.runs)

                dense_names = [n for n in ("sdpa", "flex_dense",
                                           "dense_tuned") if n in timed]
                best_dense = min(dense_names,
                                 key=lambda n: timed[n]["min_ms"])
                best_dense_med = min(dense_names,
                                     key=lambda n: timed[n]["median_ms"])
                dense_flops = 4 * B * N * N * d
                row = {
                    "mode": mode, "head_dim": d, "n_agents": N, "batch": B,
                    "peak_flops": peaks[mode],
                    "dense_search": {"best": dbest, "admitted": dok,
                                     "space": dall},
                    "sparse_search": {str(kk): {"best": sb[0], "admitted": sb[1],
                                                "space": sb[2]}
                                      for kk, sb in sbest.items()},
                    "timed": timed, "deviation": deviation,
                    "best_dense": best_dense,
                    "best_dense_ms": timed[best_dense]["min_ms"],
                    "best_dense_median": best_dense_med,
                    "best_dense_median_ms":
                        timed[best_dense_med]["median_ms"],
                    "dense_pct_peak": 100 * dense_flops / (
                        timed[best_dense]["min_ms"] * 1e-3) / peaks[mode],
                }
                rows.append(row)
                dump(args.out, device, B, peaks, args, t0, rows)
                line = " ".join(
                    f"k{kk}:{timed[f'fused_topk:{kk}']['min_ms'] / row['best_dense_ms']:.2f}"
                    for kk in ks if f"fused_topk:{kk}" in timed)
                print(f"{mode} d{d} N{N} best dense {best_dense} "
                      f"{row['best_dense_ms']:.4f} ms "
                      f"({row['dense_pct_peak']:.0f}% peak) | fused/dense "
                      f"{line} | flex_topk k1 "
                      f"{timed.get('flex_topk:1', {}).get('min_ms', float('nan')) / row['best_dense_ms']:.2f}",
                      flush=True)
                del q, k, v, refs
                torch.cuda.empty_cache()
    set_tf32(False)
    dump(args.out, device, B, peaks, args, t0, rows)
    print(f"\nwrote {args.out} ({round(time.time() - t0, 1)} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
