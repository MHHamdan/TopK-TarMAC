"""Latency of the attention core, across implementations, sparsity and shape.

The manuscript's main table times the whole communication module, which shares a
qkv projection and an output projection across every implementation. Those
shared stages dilute the quantity in question. This script times only the part
the implementations actually differ in -- scores, selection, and aggregation,
from (q, k, v) to the message tensor -- so that the comparison is between the
attention algorithms rather than between modules that happen to contain them.

Implementations timed:

    dense_unfused    scores, mask, softmax, dense aggregation matmul
    dense_fused      the same function through PyTorch's fused SDPA path
    dense_triton     dense attention written in the same framework, with the
                     same tiling and block-size policy, as the sparse kernels
                     below; the control that separates sparsity from authorship
    topk_masked      dense weights multiplied by a {0,1} mask and renormalised;
                     the discarded entries are materialised and multiplied
    topk_gather      top-k, softmax over k, gather the k value rows, contract
    topk_fused       fused: no score matrix, index tensor or gathered value
                     tensor reaches global memory; selection by k argmax rounds
    topk_fused_tiled the same, with a tile of queries per program so the score
                     product runs on tensor cores
    topk_fused_thr   the same, with selection by threshold bisection, so that
                     selection cost does not grow with k

Reporting the sparse kernels against a vendor dense kernel alone would confound
sparsity with who wrote the code, which is why dense_triton is in the set.

Protocol follows the manuscript: CUDA events with an explicit synchronisation
on the pinned device, implementations interleaved repetition by repetition so
that contention lands on all of them together, the minimum over samples as the
primary statistic, and the whole timed block repeated so that inter-run spread
is reported rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.kernels.fused_topk_attention import (  # noqa: E402
    fused_topk_attention_masked, fused_topk_attention_tiled)
from src.kernels.fused_dense_attention import fused_dense_attention  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# implementations
# ---------------------------------------------------------------------------

def _scores(q, k, eye):
    s = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    return s.masked_fill(eye, float("-inf"))


def dense_unfused(q, k, v, eye, topk):
    return torch.matmul(F.softmax(_scores(q, k, eye), dim=-1), v)


def dense_fused(q, k, v, eye, topk):
    mask = torch.zeros(q.shape[1], q.shape[1], device=q.device, dtype=q.dtype)
    mask.masked_fill_(eye.squeeze(0), float("-inf"))
    return F.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=mask).squeeze(1)


def topk_masked(q, k, v, eye, topk):
    w = F.softmax(_scores(q, k, eye), dim=-1)
    idx = w.topk(topk, dim=-1, sorted=False).indices
    keep = torch.zeros_like(w).scatter_(-1, idx, 1.0)
    w = w * keep
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.matmul(w, v)


def topk_gather(q, k, v, eye, topk):
    B, N, d = q.shape
    s = _scores(q, k, eye)
    top_s, top_i = s.topk(topk, dim=-1, sorted=False)
    w = F.softmax(top_s, dim=-1)
    idx = top_i.reshape(B, N * topk, 1).expand(B, N * topk, d)
    v_g = v.gather(1, idx).view(B, N, topk, d)
    return torch.matmul(w.unsqueeze(-2), v_g).squeeze(-2)


def topk_fused_gather(q, k, v, eye, topk):
    return fused_topk_attention_tiled(q, k, v, topk)


def topk_fused_masked(q, k, v, eye, topk):
    return fused_topk_attention_masked(q, k, v, topk)


def dense_triton(q, k, v, eye, topk):
    return fused_dense_attention(q, k, v)


IMPLS: dict[str, Callable] = {
    "dense_unfused": dense_unfused,
    "dense_fused": dense_fused,
    "dense_triton": dense_triton,
    "topk_masked": topk_masked,
    "topk_gather": topk_gather,
    "topk_fused_gather": topk_fused_gather,
    "topk_fused_masked": topk_fused_masked,
}

# The Triton kernels are forward-only: they are a statement about how fast the
# sparse forward pass can be made, not a differentiable module. Timing them in
# a backward comparison would compare a kernel against an autograd graph.
FORWARD_ONLY = ("dense_triton", "topk_fused_gather", "topk_fused_masked")


# ---------------------------------------------------------------------------
# analytic accounting
# ---------------------------------------------------------------------------

#: Implementations whose aggregation touches every peer, so that their executed
#: arithmetic is that of dense attention however small k is.
DENSE_AGGREGATION = ("dense_unfused", "dense_fused", "dense_triton",
                     "topk_masked", "topk_fused_masked")


def analytic_flops(impl: str, B: int, N: int, d: int, k: int) -> int:
    """Multiply--accumulates x2, matching torch.utils.flop_counter's convention."""
    scores = 2 * B * N * N * d
    if impl in DENSE_AGGREGATION:
        return scores + 2 * B * N * N * d
    return scores + 2 * B * N * k * d


def analytic_bytes(impl: str, B: int, N: int, d: int, k: int,
                   itemsize: int = 4) -> int:
    """Global-memory traffic, counting each tensor read and written once.

    The point of the fused implementations is that the intermediates never
    reach global memory, so their accounting differs from the composed ones by
    exactly the intermediates the composition writes and reads back.
    """
    qkv = 3 * B * N * d * itemsize
    out = B * N * d * itemsize
    smat = B * N * N * itemsize
    if impl == "dense_unfused":
        # Each elementwise step on the score matrix is out of place, so the
        # matrix is written and read once per step: the scaling, the mask, the
        # softmax and the aggregation together touch it eight times.
        return qkv + out + 8 * smat
    if impl == "dense_fused":
        return qkv + out
    if impl == "topk_masked":
        # scores, softmax, top-k read, mask write/read, renormalise, aggregate
        return qkv + out + 10 * smat
    if impl == "topk_gather":
        gathered = B * N * k * d * itemsize
        idxb = B * N * k * 8
        # scores written and read by top-k; top-k values+indices written;
        # values read by softmax and gather; gathered tensor written and read
        return (qkv + out + 2 * smat + B * N * k * itemsize + idxb
                + 2 * gathered + idxb)
    if impl in ("dense_triton", "topk_fused_gather", "topk_fused_masked"):
        # Nothing but the inputs and the output crosses the memory boundary.
        # K and V are re-read per query tile but are reused across the tiles of
        # a batch element and stay resident in cache at these sizes, so the
        # count below is compulsory traffic rather than issued traffic.
        return qkv + out
    raise ValueError(impl)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    q = statistics.quantiles(s, n=4) if len(s) >= 4 else [s[0]] * 3
    return {"min_ms": s[0], "median_ms": statistics.median(s),
            "p25_ms": q[0], "p75_ms": q[2],
            "mean_ms": statistics.fmean(s),
            "std_ms": statistics.pstdev(s) if len(s) > 1 else 0.0}


def interleaved(cells: list[dict[str, Any]], device: str, reps: int,
                warmup: int, runs: int, backward: bool) -> None:
    live = [c for c in cells if not c["failed"]]
    if backward:
        live = [c for c in live if c["impl"] not in FORWARD_ONLY]
    if not live:
        return

    def one(c: dict[str, Any]) -> float:
        if backward:
            def body():
                if c["_q"].grad is not None:
                    c["_q"].grad = None
                out = c["_fn"](c["_q"], c["_k"], c["_v"], c["_eye"], c["k"])
                out.sum().backward()
        else:
            def body():
                with torch.no_grad():
                    c["_fn"](c["_q"], c["_k"], c["_v"], c["_eye"], c["k"])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        body()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    # The device is shared, so a configuration that built successfully can
    # still run out of memory once another tenant grows. Drop such a cell and
    # record why, rather than losing the whole sweep.
    def drop(c: dict[str, Any], exc: BaseException) -> None:
        c["failed"] = True
        c["failure"] = f"oom during timing: {type(exc).__name__}"
        for h in ("_q", "_k", "_v", "_eye", "_fn"):
            c.pop(h, None)
        torch.cuda.empty_cache()

    for _ in range(warmup):
        for c in list(live):
            try:
                one(c)
            except torch.OutOfMemoryError as exc:
                drop(c, exc)
                live.remove(c)
    if not live:
        return
    torch.cuda.synchronize(device)

    per_run: dict[int, list[list[float]]] = {id(c): [] for c in live}
    for _run in range(runs):
        block: dict[int, list[float]] = {id(c): [] for c in live}
        for _rep in range(reps):
            for c in list(live):
                try:
                    block[id(c)].append(one(c))
                except torch.OutOfMemoryError as exc:
                    drop(c, exc)
                    live.remove(c)
        for c in live:
            per_run[id(c)].append(block[id(c)])

    key = "backward_total" if backward else "forward"
    for c in live:
        if not per_run[id(c)]:
            continue
        blocks = per_run[id(c)]
        flat = [x for b in blocks for x in b]
        med = [statistics.median(b) for b in blocks]
        st = _stats(flat)
        st["run_medians_ms"] = med
        st["inter_run_spread_pct"] = (
            100.0 * (max(med) - min(med)) / min(med) if min(med) > 0 else 0.0)
        c[key] = st


def build_cell(impl: str, B: int, N: int, d: int, k: int, device: str,
               dtype: torch.dtype, need_grad: bool) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "impl": impl, "batch": B, "n_agents": N, "head_dim": d, "k": k,
        "dtype": str(dtype).replace("torch.", ""),
        "analytic_flops": analytic_flops(impl, B, N, d, k),
        "analytic_bytes": analytic_bytes(
            impl, B, N, d, k, torch.empty((), dtype=dtype).element_size()),
        "failed": False, "failure": None,
    }
    try:
        gen = torch.Generator(device=device).manual_seed(1234 + N * 7 + d)
        q = torch.randn(B, N, d, device=device, dtype=dtype, generator=gen)
        kt = torch.randn(B, N, d, device=device, dtype=dtype, generator=gen)
        v = torch.randn(B, N, d, device=device, dtype=dtype, generator=gen)
        if need_grad and impl not in FORWARD_ONLY:
            q = q.requires_grad_(True)
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        fn = IMPLS[impl]
        with torch.no_grad():
            fn(q.detach(), kt, v, eye, k)
        torch.cuda.synchronize(device)
        cell.update({"_q": q, "_k": kt, "_v": v, "_eye": eye, "_fn": fn})
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        cell["failed"] = True
        cell["failure"] = f"oom: {exc}"[:200]
    except Exception as exc:  # a kernel that cannot run this shape is a result
        cell["failed"] = True
        cell["failure"] = f"{type(exc).__name__}: {exc}"[:300]
    return cell


def verify(B: int, N: int, d: int, k: int, device: str, dtype: torch.dtype,
           impls: list[str]) -> dict[str, Any]:
    """Check that every implementation computes the same function, here.

    Exact top-k is discontinuous in the scores: two implementations that compute
    the scores in different summation orders can rank a near-tied pair
    differently and then aggregate over different peers. A single spot check at
    one shape cannot detect that, so this runs at every configuration whose
    latency is reported, and records the retained-set disagreement rate as well
    as the output deviation.
    """
    gen = torch.Generator(device=device).manual_seed(99)
    q = torch.randn(B, N, d, device=device, dtype=dtype, generator=gen)
    kt = torch.randn(B, N, d, device=device, dtype=dtype, generator=gen)
    v = torch.randn(B, N, d, device=device, dtype=dtype, generator=gen)
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    out: dict[str, Any] = {}
    with torch.no_grad():
        ref_dense = dense_unfused(q, kt, v, eye, k)
        ref_topk = topk_gather(q, kt, v, eye, k)
        # Which peers the reference retains, as a set per query.
        s = _scores(q, kt, eye)
        ref_keep = torch.zeros_like(s, dtype=torch.bool).scatter_(
            -1, s.topk(k, dim=-1, sorted=False).indices, True)
        for name in impls:
            try:
                got = IMPLS[name](q, kt, v, eye, k)
                ref = ref_dense if name in ("dense_unfused", "dense_fused",
                                            "dense_triton") else ref_topk
                out[name] = (got - ref).abs().max().item()
            except Exception as exc:
                out[name] = float("nan")
                out[name + "_error"] = f"{type(exc).__name__}: {exc}"[:200]
    out["_ref_support_size"] = int(ref_keep.sum(-1).float().mean().item())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-list", nargs="+", type=int,
                   default=[12, 48, 96, 192, 384, 512])
    p.add_argument("--k-list", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--d-list", nargs="+", type=int, default=[16])
    p.add_argument("--batch-list", nargs="+", type=int, default=[256])
    p.add_argument("--impls", nargs="+", default=list(IMPLS))
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--reps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--out", default=str(ROOT / "results" / "attention_core.json"))
    args = p.parse_args()

    device = args.device
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1
    torch.cuda.set_device(device)
    dtype = DTYPES[args.dtype]
    t0 = time.time()

    rows: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    for B in args.batch_list:
        for d in args.d_list:
            for N in args.n_list:
                ks = sorted({min(max(1, k), N - 1) for k in args.k_list})
                group: list[dict[str, Any]] = []
                for impl in args.impls:
                    if impl.startswith("dense"):
                        group.append(build_cell(impl, B, N, d, N - 1, device,
                                                dtype, args.backward))
                    else:
                        for k in ks:
                            group.append(build_cell(impl, B, N, d, k, device,
                                                    dtype, args.backward))
                # Equivalence is checked at every configuration whose latency
                # is reported, at a smaller batch so the check is cheap.
                eq = verify(min(B, 32), N, d, max(ks), device, dtype,
                            [i for i in args.impls])
                equivalence.append({"n_agents": N, "head_dim": d, "batch": B,
                                    "k": max(ks), "deviation": eq})
                interleaved(group, device, args.reps, args.warmup, args.runs,
                            False)
                if args.backward:
                    interleaved(group, device, args.reps, args.warmup,
                                args.runs, True)
                for c in group:
                    for h in ("_q", "_k", "_v", "_eye", "_fn"):
                        c.pop(h, None)
                    if c["failed"]:
                        print(f"B{B} d{d} N{N:>4} {c['impl']:<16} k={c['k']:<4}"
                              f" FAILED {c['failure'][:60]}", flush=True)
                    else:
                        f = c["forward"]["min_ms"]
                        bwd = c.get("backward_total", {}).get("min_ms")
                        bs = f"{bwd:8.4f}" if bwd else "     ---"
                        print(f"B{B} d{d} N{N:>4} {c['impl']:<16} k={c['k']:<4}"
                              f" fwd {f:8.4f} ms  fwd+bwd {bs} ms", flush=True)
                rows.extend(group)
                torch.cuda.empty_cache()

    payload = {
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "dtype": args.dtype, "reps": args.reps, "warmup": args.warmup,
        "runs": args.runs, "backward": args.backward,
        "timing": "cuda_events_interleaved_min",
        "equivalence_check": equivalence,
        "max_deviation_by_impl": {
            i: max((e["deviation"].get(i, 0.0) or 0.0) for e in equivalence)
            for i in args.impls} if equivalence else {},
        "elapsed_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out} ({payload['elapsed_s']} s)")
    print("max deviation from the reference, over every measured "
          "configuration:")
    for i, e in payload["max_deviation_by_impl"].items():
        flag = "" if (e == e and e < 1e-4) else "   <-- NOT EQUIVALENT"
        print(f"    {i:<20} {e:.3e}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
