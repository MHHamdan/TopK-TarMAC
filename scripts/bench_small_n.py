"""Attention cost at the agent counts multi-agent communication actually uses.

Every latency claim elsewhere in the manuscript starts at N = 192, because below
it one Python-level call is dominated by the host launch path rather than by the
kernels it launches. This sweep measures N = 12..96 (and 192 as the anchor to
the main sweep) under two protocols, and records, per implementation, the
quantity that the launch-bound regime is governed by:

    graph        each call captured in a CUDA graph and timed by replay; host
                 launch latency is removed for every implementation alike and
                 what remains is device time (as in scripts/bench_fixed_set.py)
    eager        CUDA events around one Python-level call: the protocol of the
                 main sweep. The driver submits the start event with the first
                 launch, so this does not see the host gap before it
    eager_wall   wall clock for one eager call plus synchronize: end-to-end
                 latency of the call as a training loop would see it
    host         wall clock for the host to issue one call (no sync inside)
    graph_wall   wall clock for one graph replay plus synchronize

    launches     kernel / memset / memcpy nodes in the captured CUDA graph,
                 read through cudaGraphGetNodes, cross-checked against
                 torch.profiler's device activity for one eager call

Implementations (exact top-k unless noted):

    dense_fused      library SDPA (fused), mask built per call as in the core sweep
    dense_tuned      our dense Triton kernel at the searched tiling where one
                     exists, default tiling otherwise
    dense_unfused    scores, mask, softmax, aggregation as separate operators
    topk_masked      dense weights times a {0,1} top-k mask, renormalised
    topk_gather      composed score-then-select: scores, torch.topk, gather
    fused_stsel      fused score-then-select (fused_topk_attention_tiled)
    fixed_set        fused fixed-set kernel given the exact top-k indices free
    prescore_morton  Z-order selector + fused fixed-set kernel (different
                     retained set: a cost comparison, not the same function)

``--mode crossover`` re-measures only the fused score-then-select kernel at
small k against both dense paths, for several independent sessions (fresh
inputs, fresh graph capture) each holding several interleaved blocks, to
settle whether a crossover holds under the median as well as the minimum.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.bench_attention_core import (  # noqa: E402
    analytic_bytes, analytic_flops, dense_fused, dense_tuned, dense_unfused,
    topk_fused_gather, topk_gather, topk_masked)
from scripts.bench_fixed_set import find_set, time_group  # noqa: E402
from src.kernels.fused_prescore_attention import (  # noqa: E402
    fused_prescore_attention)
from src.kernels.prescore_selection import select_morton_window  # noqa: E402

DENSE = ("dense_fused", "dense_tuned")


# ---------------------------------------------------------------------------
# launch accounting
# ---------------------------------------------------------------------------

def _cudart() -> ctypes.CDLL | None:
    import glob
    import site
    for sp in site.getsitepackages():
        for f in glob.glob(f"{sp}/nvidia/cuda_runtime/lib/libcudart.so*"):
            return ctypes.CDLL(f)
    try:
        return ctypes.CDLL("libcudart.so")
    except OSError:
        pass
    # Builds without the nvidia-* wheels (e.g. a cluster's own torch) link a
    # libcudart outside site-packages; torch has loaded it by now.
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.split()[-1]
                if "/libcudart.so" in path:
                    return ctypes.CDLL(path)
    except OSError:
        pass
    return None


_RT = _cudart()
#: cudaGraphNodeType
_NODE_KIND = {0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "graph",
              5: "empty", 6: "wait_event", 7: "event_record",
              8: "ext_sem_signal", 9: "ext_sem_wait", 10: "mem_alloc",
              11: "mem_free", 12: "batch_memop", 13: "conditional"}


def graph_nodes(g: torch.cuda.CUDAGraph) -> dict[str, int]:
    """Count the nodes of a captured graph by type."""
    if _RT is None:
        return {}
    h = ctypes.c_void_p(g.raw_cuda_graph())
    n = ctypes.c_size_t(0)
    assert _RT.cudaGraphGetNodes(h, None, ctypes.byref(n)) == 0
    nodes = (ctypes.c_void_p * n.value)()
    assert _RT.cudaGraphGetNodes(h, nodes, ctypes.byref(n)) == 0
    out: dict[str, int] = {}
    for i in range(n.value):
        t = ctypes.c_int(0)
        assert _RT.cudaGraphNodeGetType(ctypes.c_void_p(nodes[i]),
                                        ctypes.byref(t)) == 0
        k = _NODE_KIND.get(t.value, str(t.value))
        out[k] = out.get(k, 0) + 1
    return out


def profiler_launches(f: Callable[[], Any]) -> dict[str, int]:
    """Device activities and host launch API calls for one eager call."""
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        with torch.no_grad():
            f()
        torch.cuda.synchronize()
    dev = api = 0
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            dev += 1
        elif ("LaunchKernel" in e.name or e.name.startswith("cuLaunch")
              or "Memset" in e.name or "MemcpyAsync" in e.name):
            api += 1
    return {"device_ops": dev, "launch_api_calls": api}


def as_graphs(fns: dict[str, Callable[[], Any]], device: str
              ) -> tuple[dict[str, Callable[[], Any]], dict[str, dict]]:
    """bench_fixed_set.as_graphs, keeping each graph to count its nodes."""
    out: dict[str, Callable[[], Any]] = {}
    nodes: dict[str, dict] = {}
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side), torch.no_grad():
        for f in fns.values():
            for _ in range(3):
                f()
    torch.cuda.current_stream(device).wait_stream(side)
    keep = []
    for n, f in fns.items():
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.no_grad(), torch.cuda.graph(g):
            f()
        nodes[n] = graph_nodes(g)
        g.instantiate()
        keep.append(g)
        out[n] = g.replay
    torch.cuda.synchronize(device)
    out["__keep__"] = keep  # type: ignore[assignment]
    return out, nodes


def host_times(fns: dict[str, Callable[[], Any]], device: str, reps: int,
               warmup: int) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Wall time to issue one call, and to issue and complete it (call plus
    synchronize), interleaved; the device drains between calls.

    The second is the end-to-end latency of one eager call. CUDA events around
    the call do not capture it: the driver submits the start event together
    with the first launch, so the host gap between them is not timed.
    """
    names = list(fns)
    issue: dict[str, list[float]] = {n: [] for n in names}
    wall: dict[str, list[float]] = {n: [] for n in names}
    for i in range(warmup + reps):
        for n in names:
            torch.cuda.synchronize(device)
            t = time.perf_counter()
            with torch.no_grad():
                fns[n]()
            t1 = time.perf_counter()
            torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            if i >= warmup:
                issue[n].append((t1 - t) * 1e3)
                wall[n].append((t2 - t) * 1e3)
    torch.cuda.synchronize(device)
    return issue, wall


# ---------------------------------------------------------------------------
# analytic accounting for the paths bench_attention_core does not cover
# ---------------------------------------------------------------------------

def flops(impl: str, B: int, N: int, d: int, k: int) -> int:
    if impl in ("fixed_set", "prescore_morton"):
        return 4 * B * N * k * d  # Nk scores + Nk aggregation; selector ~0 FLOP
    if impl == "fused_stsel":
        return analytic_flops("topk_fused_gather", B, N, d, k)
    return analytic_flops(impl, B, N, d, k)


def nbytes(impl: str, B: int, N: int, d: int, k: int) -> int:
    io = 4 * B * N * d * 4
    idx = B * N * k * 8
    if impl == "fixed_set":
        return io + idx
    if impl == "prescore_morton":
        # pos read; min/max, code, argsort keys+values, rank scatter: ~6 (B,N)
        # int64 tensors written and read; index tensor written and read.
        return io + B * N * 2 * 4 + 12 * B * N * 8 + 2 * idx
    if impl == "fused_stsel":
        return analytic_bytes("topk_fused_gather", B, N, d, k)
    return analytic_bytes(impl, B, N, d, k)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def ratio_stats(x_runs: list[list[float]], dense_runs: dict[str, list[list[float]]],
                n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    x = np.concatenate([np.asarray(b) for b in x_runs])
    flat = {n: np.concatenate([np.asarray(b) for b in s])
            for n, s in dense_runs.items()}
    best = min(flat, key=lambda n: flat[n].min())
    best_med = min(flat, key=lambda n: np.median(flat[n]))
    dense_rep = np.minimum.reduce(list(flat.values()))
    paired = x / dense_rep
    boot = np.median(paired[rng.integers(0, len(paired),
                                         (n_boot, len(paired)))], axis=1)
    per_run = [float(np.median(np.asarray(a) / np.minimum.reduce(
        [np.asarray(dense_runs[d][i]) for d in dense_runs])))
        for i, a in enumerate(x_runs)]
    return {
        "best_dense_by_min": best, "best_dense_by_median": best_med,
        "ratio_of_minima": float(x.min() / flat[best].min()),
        "ratio_of_medians": float(np.median(x) / np.median(flat[best_med])),
        "paired_median": float(np.median(paired)),
        "paired_median_ci95": [float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))],
        "per_run_paired_median": per_run,
    }


def desc(runs: list[list[float]]) -> dict[str, Any]:
    x = np.concatenate([np.asarray(b) for b in runs])
    med = [float(np.median(b)) for b in runs]
    return {"min_ms": float(x.min()), "median_ms": float(np.median(x)),
            "p25_ms": float(np.percentile(x, 25)),
            "p75_ms": float(np.percentile(x, 75)),
            "run_medians_ms": med,
            "inter_run_spread_pct": 100 * (max(med) - min(med)) / min(med)}


def _inputs(B: int, N: int, d: int, device: str, seed: int):
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, N, d, device=device, generator=gen)
    kt = torch.randn(B, N, d, device=device, generator=gen)
    v = torch.randn(B, N, d, device=device, generator=gen)
    pos = torch.rand(B, N, 2, device=device, generator=gen)
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    return q, kt, v, pos, eye


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def small_n_group(B: int, N: int, d: int, ks: list[int], args, rng
                  ) -> dict[str, Any]:
    device = args.device
    q, kt, v, pos, eye = _inputs(B, N, d, device, 1234 + N * 7 + d + B)
    fns: dict[str, Callable[[], Any]] = {
        "dense_fused": lambda: dense_fused(q, kt, v, eye, 1),
        "dense_tuned": lambda: dense_tuned(q, kt, v, eye, 1),
        "dense_unfused": lambda: dense_unfused(q, kt, v, eye, 1),
    }
    eq: dict[int, dict[str, float]] = {}
    for k in ks:
        fns[f"topk_masked@{k}"] = lambda k=k: topk_masked(q, kt, v, eye, k)
        fns[f"topk_gather@{k}"] = lambda k=k: topk_gather(q, kt, v, eye, k)
        fns[f"fused_stsel@{k}"] = lambda k=k: topk_fused_gather(q, kt, v, eye, k)
        idx = find_set(q, kt, v, eye, k)
        fns[f"fixed_set@{k}"] = lambda i=idx: fused_prescore_attention(q, kt, v, i)
        fns[f"prescore_morton@{k}"] = (
            lambda k=k: fused_prescore_attention(q, kt, v,
                                                 select_morton_window(pos, k)))
        with torch.no_grad():
            ref = topk_gather(q, kt, v, eye, k)
            eq[k] = {n: float((fns[f"{n}@{k}"]() - ref).abs().max())
                     for n in ("topk_masked", "fused_stsel", "fixed_set")}
    for f in fns.values():
        with torch.no_grad():
            f()
    torch.cuda.synchronize(device)

    launches = {n: profiler_launches(f) for n, f in fns.items()}
    graphs, nodes = as_graphs(fns, device)
    keep = graphs.pop("__keep__")
    raw = {
        "graph": time_group(graphs, device, args.reps, args.warmup, args.runs),
        "eager": time_group(fns, device, args.reps, args.warmup, args.runs),
    }
    host, wall = host_times(fns, device, args.reps * args.runs, args.warmup)
    graph_wall = host_times(graphs, device, args.reps * args.runs,
                            args.warmup)[1]
    del keep, graphs

    impls: dict[str, Any] = {}
    for n in fns:
        base, _, kk = n.partition("@")
        k = int(kk) if kk else None
        kf = k if k is not None else 1
        rec: dict[str, Any] = {
            "impl": base, "k": k,
            "graph_nodes": nodes[n], **launches[n],
            "launches": sum(c for t, c in nodes[n].items()
                            if t in ("kernel", "memset", "memcpy")),
            "flops": flops(base, B, N, d, kf),
            "bytes": nbytes(base, B, N, d, kf),
            "host_issue_ms": {"min": float(np.min(host[n])),
                              "median": float(np.median(host[n]))},
            "eager_wall_ms": {"min": float(np.min(wall[n])),
                              "median": float(np.median(wall[n]))},
            "graph_wall_ms": {"min": float(np.min(graph_wall[n])),
                              "median": float(np.median(graph_wall[n]))},
        }
        if base not in DENSE:
            for key, src in (("eager_wall", wall), ("graph_wall", graph_wall)):
                rec[key + "_vs_best_dense"] = ratio_stats(
                    [src[n]], {dn: [src[dn]] for dn in DENSE}, args.n_boot, rng)
        for mode in ("graph", "eager"):
            rec[mode] = desc(raw[mode][n])
            if base not in DENSE:
                rec[mode]["vs_best_dense"] = ratio_stats(
                    raw[mode][n], {dn: raw[mode][dn] for dn in DENSE},
                    args.n_boot, rng)
            rec[mode]["raw_ms"] = raw[mode][n]
        impls[n] = rec
        g, e = rec["graph"], rec["eager"]
        print(f"B{B:<5}N{N:<4}{n:<22} L={rec['launches']:<3}"
              f" graph {g['min_ms']*1e3:7.1f}/{g['median_ms']*1e3:7.1f}us"
              f" eager {e['min_ms']*1e3:7.1f}/{e['median_ms']*1e3:7.1f}us"
              f" host {rec['host_issue_ms']['median']*1e3:6.1f}us"
              f" wall {rec['eager_wall_ms']['median']*1e3:6.1f}us"
              + (f"  r_g {g['vs_best_dense']['ratio_of_minima']:.2f}"
                 f"/{g['vs_best_dense']['ratio_of_medians']:.2f}"
                 if "vs_best_dense" in g else ""), flush=True)
    del q, kt, v, pos, eye, fns
    torch.cuda.empty_cache()
    return {"batch": B, "n_agents": N, "head_dim": d, "ks": ks,
            "dtype": "float32", "max_abs_dev_vs_topk_gather": eq,
            "impls": impls}


def crossover_group(N: int, d: int, ks: list[int], args, rng
                    ) -> list[dict[str, Any]]:
    """Several sessions, each with fresh inputs and fresh graphs."""
    device, B = args.device, args.batch_list[0]
    sessions: dict[str, dict[str, list[list[float]]]] = {}
    for s in range(args.sessions):
        q, kt, v, _pos, eye = _inputs(B, N, d, device,
                                      1234 + N * 7 + d + 1000 * s)
        fns: dict[str, Callable[[], Any]] = {
            "dense_fused": lambda: dense_fused(q, kt, v, eye, 1),
            "dense_tuned": lambda: dense_tuned(q, kt, v, eye, 1)}
        for k in ks:
            fns[f"fused_stsel@{k}"] = (
                lambda k=k: topk_fused_gather(q, kt, v, eye, k))
        for f in fns.values():
            with torch.no_grad():
                f()
        torch.cuda.synchronize(device)
        for mode in args.cross_modes:
            if mode == "graph":
                timed, _ = as_graphs(fns, device)
                keep = timed.pop("__keep__")
            else:
                timed, keep = fns, None
            raw = time_group(timed, device, args.reps, args.warmup, args.runs)
            for n, r in raw.items():
                sessions.setdefault(f"{mode}|{n}", {}).setdefault(
                    "runs", []).extend(r)
            del keep, timed
        del q, kt, v, eye, fns
        torch.cuda.empty_cache()

    out = []
    nb = args.runs  # blocks per session
    for mode in args.cross_modes:
        dense = {dn: sessions[f"{mode}|{dn}"]["runs"] for dn in DENSE}
        for k in ks:
            x = sessions[f"{mode}|fused_stsel@{k}"]["runs"]
            pooled = ratio_stats(x, dense, args.n_boot, rng)
            per_session = [ratio_stats(
                x[i * nb:(i + 1) * nb],
                {dn: r[i * nb:(i + 1) * nb] for dn, r in dense.items()},
                args.n_boot, rng) for i in range(args.sessions)]
            out.append({
                "mode": mode, "n_agents": N, "head_dim": d, "k": k,
                "batch": B, "sessions": args.sessions, "blocks_per_session": nb,
                "reps_per_block": args.reps,
                "pooled": pooled, "per_session": per_session,
                "dense": {dn: desc(r) for dn, r in dense.items()},
                "fused_stsel": desc(x),
                "raw_ms": {"fused_stsel": x, **dense},
            })
            p = pooled
            print(f"{mode:<5} d={d:<3} N={N:<4} k={k}  min {p['ratio_of_minima']:.3f}"
                  f"  med {p['ratio_of_medians']:.3f}  paired "
                  f"{p['paired_median']:.3f} [{p['paired_median_ci95'][0]:.3f},"
                  f"{p['paired_median_ci95'][1]:.3f}]  sessions(min/med) "
                  + " ".join(f"{q['ratio_of_minima']:.2f}/{q['ratio_of_medians']:.2f}"
                             for q in per_session), flush=True)
    return out


# ---------------------------------------------------------------------------
# analysis: what predicts latency at N <= 96, and does the ordering carry over
# ---------------------------------------------------------------------------

ORDER_IMPLS = ("dense_fused", "dense_tuned", "dense_unfused", "topk_masked",
               "topk_gather", "fused_stsel", "fixed_set", "prescore_morton")


def analyse(path: Path, n_max: int = 96, anchor: int = 192) -> dict[str, Any]:
    from scipy import stats as st
    data = json.loads(path.read_text())
    rows = []
    for g in data["groups"]:
        for name, r in g["impls"].items():
            rows.append({
                "B": g["batch"], "N": g["n_agents"], "impl": r["impl"],
                "k": r["k"], "launches": r["launches"], "flops": r["flops"],
                "bytes": r["bytes"],
                "eager_wall_min": r["eager_wall_ms"]["min"],
                "host_issue_med": r["host_issue_ms"]["median"],
                "eager_event_med": r["eager"]["median_ms"],
                "graph_med": r["graph"]["median_ms"],
                "graph_min": r["graph"]["min_ms"]})
    small = [r for r in rows if r["N"] <= n_max]
    targets = ("eager_wall_min", "host_issue_med", "eager_event_med",
               "graph_med")
    preds = ("launches", "flops", "bytes")
    out: dict[str, Any] = {"n_rows": len(small), "predictors": {}}
    for t in targets:
        out["predictors"][t] = {}
        for x in preds:
            xs = np.array([r[x] for r in small], float)
            ys = np.array([r[t] for r in small], float)
            fit = st.linregress(xs, ys)
            per_group = []
            for key in sorted({(r["B"], r["N"]) for r in small}):
                sub = [r for r in small if (r["B"], r["N"]) == key]
                rho = st.spearmanr([r[x] for r in sub], [r[t] for r in sub])
                per_group.append(float(rho.statistic))
            out["predictors"][t][x] = {
                "pooled_spearman": float(st.spearmanr(xs, ys).statistic),
                "pooled_linear_r2": float(fit.rvalue ** 2),
                "slope_ms_per_unit": float(fit.slope),
                "intercept_ms": float(fit.intercept),
                "within_group_spearman_median": float(np.median(per_group)),
                "within_group_spearman_min": float(np.min(per_group)),
            }
    # Ordering of implementations under graph replay vs the anchor size.
    order = []
    for B in sorted({r["B"] for r in rows}):
        anc = {(r["impl"], r["k"]): r["graph_med"] for r in rows
               if r["B"] == B and r["N"] == anchor}
        if not anc:
            continue
        for N in sorted({r["N"] for r in rows if r["B"] == B and r["N"] < anchor}):
            for k in (1, 2, 4):
                cur = {r["impl"]: r["graph_med"] for r in rows
                       if r["B"] == B and r["N"] == N and r["k"] in (k, None)}
                ref = {i: anc.get((i, k), anc.get((i, None)))
                       for i in ORDER_IMPLS}
                names = [i for i in ORDER_IMPLS if i in cur and ref.get(i)]
                tau = st.kendalltau([cur[i] for i in names],
                                    [ref[i] for i in names]).statistic
                rank = lambda d: sorted(names, key=lambda i: d[i])
                order.append({"batch": B, "n_agents": N, "k": k,
                              "kendall_tau_vs_anchor": float(tau),
                              "order": rank(cur), "anchor_order": rank(ref)})
    out["graph_order_vs_anchor"] = order
    data["analysis"] = out
    path.write_text(json.dumps(data, indent=1))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="small_n", choices=["small_n", "crossover"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-list", nargs="+", type=int, default=[12, 24, 48, 96, 192])
    p.add_argument("--k-list", nargs="+", default=["1", "2", "4", "N/4"])
    p.add_argument("--d-list", nargs="+", type=int, default=[16])
    p.add_argument("--batch-list", nargs="+", type=int, default=[256])
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--cross-modes", nargs="+", default=["graph", "eager"])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default=None)
    p.add_argument("--analyse", default=None,
                   help="only analyse an existing small_n.json, no GPU")
    p.add_argument("--append", action="store_true",
                   help="merge groups into an existing output file")
    args = p.parse_args()
    if args.analyse:
        res = analyse(Path(args.analyse))
        print(json.dumps(res["predictors"], indent=1))
        for o in res["graph_order_vs_anchor"]:
            print(o["batch"], o["n_agents"], o["k"],
                  round(o["kendall_tau_vs_anchor"], 2), o["order"])
        return 0

    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(0)
    t0 = time.time()
    groups: list[dict[str, Any]] = []
    for B in args.batch_list:
        for d in args.d_list:
            for N in args.n_list:
                ks = sorted({max(1, N // 4) if k == "N/4" else int(k)
                             for k in args.k_list} & set(range(1, N)))
                if args.mode == "small_n":
                    groups.append(small_n_group(B, N, d, ks, args, rng))
                else:
                    groups.extend(crossover_group(N, d, ks, args, rng))
    out = Path(args.out or ROOT / "results" / (
        "small_n.json" if args.mode == "small_n" else "crossover_graph.json"))
    meta = {"script": "scripts/bench_small_n.py", "args": vars(args),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(args.device),
            "python": platform.python_version(),
            "elapsed_s": time.time() - t0}
    payload = {"meta": [meta], "groups": groups}
    if args.append and out.exists():
        old = json.loads(out.read_text())
        payload = {"meta": old["meta"] + [meta],
                   "groups": old["groups"] + groups}
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out} in {meta['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
