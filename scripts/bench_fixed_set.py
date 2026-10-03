"""Hold the retained set fixed and change only how it is found.

The pre-score selectors of Section 4.6 retain different peers from top-k, so
their comparison with score-then-select mixes the cost of selection with its
quality. This sweep removes the second factor. Every top-k implementation below
computes exact top-k attention over the same retained set, the k largest scores
of each query, and they differ only in how that set is reached:

    fused_stsel     one kernel: every score, rank, aggregate over the k kept
    find_set        every score through cuBLAS, then torch.topk; indices only
    two_stage       find_set followed by the fused fixed-set kernel
    fixed_set       the fused fixed-set kernel given the indices for free

``fixed_set`` is what any selector that recovered the exact top-k set at zero
cost would pay for the attention stage, so the gap between it and
``fused_stsel`` is the price of finding the set by scoring. Every ratio is
against the better of the library dense path and the searched dense kernel,
timed in the same interleaved group.

Raw per-repetition samples are stored, so that ratios can be recomputed under
any estimator and paired bootstrap intervals formed without re-running.

Each group is timed twice. ``eager`` is the protocol of every other sweep in the
paper: CUDA events around one Python-level call. When the host launch path is
slower than the kernel, that protocol measures the launch, and the kernels here
run for a few microseconds. ``graph`` captures each call in a CUDA graph and
times its replay, which removes host launch latency for every implementation
alike and leaves the device time.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.bench_attention_core import (  # noqa: E402
    _scores, dense_fused, dense_tuned, topk_fused_gather, topk_gather)
from src.kernels.fused_prescore_attention import (  # noqa: E402
    fused_prescore_attention)


def find_set(q, k, v, eye, topk):
    return _scores(q, k, eye).topk(topk, dim=-1, sorted=False).indices


def two_stage(q, k, v, eye, topk):
    return fused_prescore_attention(q, k, v, find_set(q, k, v, eye, topk))


TOPK_IMPLS: dict[str, Callable] = {
    "fused_stsel": topk_fused_gather,
    "find_set": find_set,
    "two_stage": two_stage,
}
DENSE_IMPLS: dict[str, Callable] = {
    "dense_fused": dense_fused,
    "dense_tuned": dense_tuned,
}


def verify(q, kt, v, eye, k: int) -> dict[str, Any]:
    """Every top-k path against the composed reference, on the same inputs.

    Also checks that the fixed-set kernel given the reference's own indices
    reproduces the reference, which is what makes ``fixed_set`` the same
    function rather than an approximation of it.
    """
    with torch.no_grad():
        ref = topk_gather(q, kt, v, eye, k)
        s = _scores(q, kt, eye)
        idx = s.topk(k, dim=-1, sorted=False).indices
        srt = s.sort(dim=-1, descending=True).values
        ties = int(((srt[..., k - 1] - srt[..., k]) == 0).sum())
        outs = {
            "fused_stsel": topk_fused_gather(q, kt, v, eye, k),
            "two_stage": two_stage(q, kt, v, eye, k),
            "fixed_set": fused_prescore_attention(q, kt, v, idx),
        }
        res: dict[str, Any] = {"rows": q.shape[0] * q.shape[1],
                               "rows_with_exact_tie_at_k": ties}
        scale = ref.abs().max().item()
        for name, got in outs.items():
            err = (got - ref).abs()
            res[name] = {
                "max_abs": err.max().item(),
                "mean_abs": err.mean().item(),
                "max_rel": err.max().item() / scale,
                "diverging_rows": int((err.amax(-1) > 1e-4).sum()),
            }
    return res


def time_group(fns: dict[str, Callable[[], Any]], device: str, reps: int,
               warmup: int, runs: int) -> dict[str, list[list[float]]]:
    """Interleave every function repetition by repetition; keep raw samples."""
    names = list(fns)

    def one(name: str) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            fns[name]()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(warmup):
        for n in names:
            one(n)
    torch.cuda.synchronize(device)
    out: dict[str, list[list[float]]] = {n: [] for n in names}
    for _r in range(runs):
        block: dict[str, list[float]] = {n: [] for n in names}
        for _i in range(reps):
            for n in names:
                block[n].append(one(n))
        for n in names:
            out[n].append(block[n])
    return out


def as_graphs(fns: dict[str, Callable[[], Any]], device: str
              ) -> dict[str, Callable[[], Any]]:
    """Capture each call in its own CUDA graph; return the replays."""
    out: dict[str, Callable[[], Any]] = {}
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side), torch.no_grad():
        for f in fns.values():
            for _ in range(3):
                f()
    torch.cuda.current_stream(device).wait_stream(side)
    for n, f in fns.items():
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            f()
        out[n] = g.replay
    torch.cuda.synchronize(device)
    return out


def summarise(samples: dict[str, list[list[float]]], n_boot: int,
              rng: np.random.Generator) -> dict[str, Any]:
    """Minimum, median and per-run medians for each implementation, and each
    top-k path's ratio against the best dense one under three estimators.

    The paired ratio pairs repetition i of an implementation with repetition i
    of the dense paths, which ran in the same interleaved round. Its bootstrap
    interval resamples repetitions with replacement; it describes the
    sampling spread of this session and not variation between sessions, which
    the per-run medians show.
    """
    flat = {n: np.concatenate([np.asarray(b) for b in s])
            for n, s in samples.items()}
    dense_names = [n for n in flat if n in DENSE_IMPLS]
    best = min(dense_names, key=lambda n: flat[n].min())
    dense_rep = np.minimum.reduce([flat[n] for n in dense_names])
    stats: dict[str, Any] = {}
    for n, x in flat.items():
        med_runs = [float(np.median(b)) for b in samples[n]]
        stats[n] = {
            "min_ms": float(x.min()), "median_ms": float(np.median(x)),
            "p25_ms": float(np.percentile(x, 25)),
            "p75_ms": float(np.percentile(x, 75)),
            "run_medians_ms": med_runs,
            "inter_run_spread_pct": 100.0 * (max(med_runs) - min(med_runs))
            / min(med_runs),
        }
    ratios: dict[str, Any] = {}
    for n, x in flat.items():
        if n in DENSE_IMPLS:
            continue
        paired = x / dense_rep
        boot = np.median(paired[rng.integers(0, len(paired),
                                             (n_boot, len(paired)))], axis=1)
        run_ratios = [float(np.median(np.asarray(a) / np.minimum.reduce(
            [np.asarray(samples[d][i]) for d in dense_names])))
            for i, a in enumerate(samples[n])]
        ratios[n] = {
            "ratio_of_minima": float(x.min() / flat[best].min()),
            "ratio_of_medians": float(np.median(x) / np.median(flat[best])),
            "paired_median": float(np.median(paired)),
            "paired_median_ci95": [float(np.percentile(boot, 2.5)),
                                   float(np.percentile(boot, 97.5))],
            "per_run_paired_median": run_ratios,
        }
    return {"best_dense": best, "stats": stats, "ratios": ratios}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-list", nargs="+", type=int, default=[96, 192, 384, 512])
    p.add_argument("--k-list", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--d-list", nargs="+", type=int, default=[16, 32])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--modes", nargs="+", default=["eager", "graph"],
                   choices=["eager", "graph"])
    p.add_argument("--out", default=str(ROOT / "results" / "fixed_set.json"))
    args = p.parse_args()

    device, B = args.device, args.batch
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(0)
    t0 = time.time()
    groups: list[dict[str, Any]] = []
    for d in args.d_list:
        for N in args.n_list:
            gen = torch.Generator(device=device).manual_seed(1234 + N * 7 + d)
            q = torch.randn(B, N, d, device=device, generator=gen)
            kt = torch.randn(B, N, d, device=device, generator=gen)
            v = torch.randn(B, N, d, device=device, generator=gen)
            eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
            # One interleaved group per (d, N) holds every k, as in the core
            # sweep, so a ratio never spans two groups.
            fns: dict[str, Callable[[], Any]] = {
                n: (lambda f=f: f(q, kt, v, eye, 1))
                for n, f in DENSE_IMPLS.items()}
            checks: dict[int, Any] = {}
            ks = [k for k in args.k_list if k < N]
            for k in ks:
                for n, f in TOPK_IMPLS.items():
                    fns[f"{n}@{k}"] = (lambda f=f, k=k: f(q, kt, v, eye, k))
                idx = find_set(q, kt, v, eye, k)
                fns[f"fixed_set@{k}"] = (
                    lambda i=idx: fused_prescore_attention(q, kt, v, i))
                checks[k] = verify(q, kt, v, eye, k)
            for f in fns.values():  # compile every kernel outside the timing
                with torch.no_grad():
                    f()
            torch.cuda.synchronize(device)
            for mode in args.modes:
                timed = fns if mode == "eager" else as_graphs(fns, device)
                raw = time_group(timed, device, args.reps, args.warmup,
                                 args.runs)
                for k in ks:
                    sub = {n: raw[n] for n in DENSE_IMPLS}
                    sub.update({n.split("@")[0]: raw[n] for n in raw
                                if n.endswith(f"@{k}")})
                    g = summarise(sub, args.n_boot, rng)
                    g.update({"mode": mode, "n_agents": N, "k": k,
                              "head_dim": d, "batch": B, "dtype": "float32",
                              "equivalence": checks[k], "raw_ms": sub})
                    groups.append(g)
                    r = g["ratios"]
                    print(f"{mode:<5} d={d:<3} N={N:<4} k={k:<3} "
                          + " ".join(f"{n} {r[n]['ratio_of_minima']:.2f}"
                                     for n in ("fused_stsel", "two_stage",
                                               "fixed_set", "find_set")),
                          flush=True)
                del timed, raw
            del q, kt, v, fns
            torch.cuda.empty_cache()

    meta = {
        "script": "scripts/bench_fixed_set.py", "args": vars(args),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "elapsed_s": time.time() - t0,
        "estimators": "ratio_of_minima is primary, as elsewhere in the paper; "
                      "paired_median pairs repetition i with the best dense "
                      "path's repetition i in the same interleaved round",
    }
    Path(args.out).write_text(json.dumps({"meta": meta, "groups": groups},
                                         indent=1))
    print(f"wrote {args.out} in {meta['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
