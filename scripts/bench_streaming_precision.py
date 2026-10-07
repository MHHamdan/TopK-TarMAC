"""The precision study (scripts/bench_precision.py) for the streaming top-k kernel.

In fp32 with ieee products the streaming kernel of
src/kernels/streaming_topk_attention.py is faster than the best dense kernel at
one and two retained peers and at parity at four (results/streaming_topk.json).
That format keeps dense attention off the tensor cores. This sweep asks whether
the result survives once dense attention gets them back:

    tf32   fp32 storage, TF32 products on the tensor cores for every kernel
           (Triton input_precision="tf32"; PyTorch's matmul TF32 switch on)
    bf16   bf16 storage and products, fp32 accumulation, bf16 output

Formats are built exactly as bench_precision.py builds them (its MODES, seeds,
float64 references and row-deviation criterion are imported, not copied).

Per (format, d, N) one interleaved group holds

    sdpa, flex_dense, dense_tuned    the dense baselines of results/precision.json;
                                     dense_tuned re-searched here per format over
                                     bench_precision.py's dense space
    streaming                        at the per-format tiling found by
                                     scripts/autotune_streaming.py --mode
                                     (results/autotune_streaming_precision.json)
    fused_resident                   the resident-row kernel at the per-format
                                     tiling results/precision.json found, for
                                     context (only where that search admitted one)

timed eagerly and as CUDA-graph replays, with raw samples stored and ratios
against the best dense path under ratio-of-minima and paired-median estimators
(scripts/bench_fixed_set.py's protocol).

Retained set: the streaming kernel's indices are compared with the float64
top-k of the same rounded inputs, on the timed inputs and on extra draws. A
configuration with more than ROW_TOL (0.1%) of rows retaining a different set is
flagged ``excluded``, the paper's criterion; it is still timed and reported.

Writes results/streaming_precision.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import triton
from torch.nn.attention.flex_attention import create_block_mask

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import bench_precision as bp  # noqa: E402
from scripts.bench_fixed_set import as_graphs, time_group  # noqa: E402
from src.kernels.streaming_topk_attention import (  # noqa: E402
    streaming_topk_attention)

DENSE = ("sdpa", "flex_dense", "dense_tuned")
TOL = {"tf32": 2e-2, "bf16": 5e-2}


def summarise(samples: dict[str, list[list[float]]], n_boot: int,
              rng: np.random.Generator) -> dict[str, Any]:
    """bench_fixed_set.summarise with this sweep's dense set."""
    flat = {n: np.concatenate([np.asarray(b) for b in s])
            for n, s in samples.items()}
    dense_names = [n for n in flat if n in DENSE]
    best = min(dense_names, key=lambda n: flat[n].min())
    best_med = min(dense_names, key=lambda n: np.median(flat[n]))
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
        if n in DENSE:
            continue
        paired = x / dense_rep
        boot = np.median(paired[rng.integers(0, len(paired),
                                             (n_boot, len(paired)))], axis=1)
        ratios[n] = {
            "ratio_of_minima": float(x.min() / flat[best].min()),
            "ratio_of_medians": float(np.median(x) / np.median(flat[best_med])),
            "paired_median": float(np.median(paired)),
            "paired_median_ci95": [float(np.percentile(boot, 2.5)),
                                   float(np.percentile(boot, 97.5))],
            "per_run_paired_median": [
                float(np.median(np.asarray(a) / np.minimum.reduce(
                    [np.asarray(samples[d][i]) for d in dense_names])))
                for i, a in enumerate(samples[n])],
        }
    return {"best_dense": best, "best_dense_median": best_med,
            "stats": stats, "ratios": ratios}


def stream_cfgs(path: Path) -> dict[tuple, dict]:
    out = {}
    if path.exists():
        for r in json.loads(path.read_text())["rows"]:
            if r.get("best"):
                out[(r["mode"], r["head_dim"], r["n_agents"], r["k"],
                     r["batch"])] = r["best"]["cfg"]
    return out


def resident_cfgs() -> dict[tuple, dict]:
    out = {}
    f = ROOT / "results" / "precision.json"
    if f.exists():
        pay = json.loads(f.read_text())
        for r in pay["rows"]:
            for kk, s in r["sparse_search"].items():
                if s["best"]:
                    out[(r["mode"], r["head_dim"], r["n_agents"], int(kk),
                         r["batch"])] = s["best"]["cfg"]
    return out


def fp32_cfg(d: int, N: int, kk: int, B: int) -> dict:
    from src.kernels.streaming_topk_attention import _tuned_cfg
    cfg = dict(_tuned_cfg(d, N, kk, B) or {})
    return {"BLOCK_M": 64, "BLOCK_N": 16, "num_warps": 2, "num_stages": 2,
            "BITONIC": True, "EARLY_EXIT": False, **cfg}


def stream(q, k, v, kk, cfg, prec, return_indices=False):
    return streaming_topk_attention(
        q, k, v, kk, block_m=cfg["BLOCK_M"], block_n=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        bitonic=cfg["BITONIC"], early_exit=cfg["EARLY_EXIT"], tuned=False,
        precision=prec, return_indices=return_indices)


def set_check(q, k, v, kk, cfg, prec, tol) -> dict[str, Any]:
    """Retained set and output against the float64 reference of these
    (already rounded) inputs."""
    N = q.shape[1]
    ref = bp.references(q, k, v, [kk])[kk]
    out, idx = stream(q, k, v, kk, cfg, prec, return_indices=True)
    eye = torch.eye(N, device=q.device, dtype=torch.bool)
    s = (q.double() @ k.double().transpose(-1, -2)).masked_fill(
        eye, float("-inf"))
    ri = s.topk(kk, dim=-1).indices.sort(-1).values
    differ = (idx.long().sort(-1).values != ri).any(-1)
    srt = s.sort(dim=-1, descending=True).values
    dev, frac = bp.disagreement(out, ref, tol)
    return {"rows": differ.numel(),
            "set_mismatch_rows": int(differ.sum()),
            "set_mismatch_frac": float(differ.float().mean()),
            "rows_exact_tie_at_k_fp64": int(
                (srt[..., kk - 1] == srt[..., kk]).sum()),
            "max_abs_vs_fp64": dev, "rows_deviating_frac": frac}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--modes", nargs="+", default=["tf32", "bf16"],
                   choices=["tf32", "bf16"])
    p.add_argument("--n-list", nargs="+", type=int, default=[192, 384, 512])
    p.add_argument("--k-list", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--d-list", nargs="+", type=int, default=[16, 32])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--search-reps", type=int, default=30)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--extra-seeds", type=int, default=2)
    p.add_argument("--tunings", default=str(
        ROOT / "results" / "autotune_streaming_precision.json"))
    p.add_argument("--tunings-unadmitted", default=str(
        ROOT / "results" / "autotune_streaming_precision_unadmitted.json"))
    p.add_argument("--out", default=str(
        ROOT / "results" / "streaming_precision.json"))
    args = p.parse_args()

    device, B = args.device, args.batch
    torch.cuda.set_device(device)
    rng = np.random.default_rng(0)
    scfg, rcfg = stream_cfgs(Path(args.tunings)), resident_cfgs()
    ucfg = stream_cfgs(Path(args.tunings_unadmitted))
    t0 = time.time()
    groups: list[dict[str, Any]] = []
    dense_search: list[dict[str, Any]] = []

    def dump():
        meta = {"script": "scripts/bench_streaming_precision.py",
                "args": vars(args), "torch": torch.__version__,
                "cuda": torch.version.cuda, "triton": triton.__version__,
                "device_name": torch.cuda.get_device_name(device),
                "python": platform.python_version(),
                "elapsed_s": time.time() - t0, "row_tol": bp.ROW_TOL,
                "tolerance": TOL,
                "estimators": "ratio_of_minima is primary; paired_median "
                              "pairs repetition i with the fastest dense path "
                              "of the same interleaved round"}
        Path(args.out).write_text(json.dumps(
            {"meta": meta, "dense_search": dense_search, "groups": groups},
            indent=1))

    for mode in args.modes:
        dtype, prec, tf32 = bp.MODES[mode]
        bp.set_tf32(tf32)
        tol = TOL[mode]
        for d in args.d_list:
            for N in args.n_list:
                gen = torch.Generator(device=device).manual_seed(77 + N + d)
                q, kt, v = (torch.randn(B, N, d, device=device, generator=gen)
                            .to(dtype) for _ in range(3))
                eye = torch.eye(N, device=device, dtype=torch.bool)
                ks = [kk for kk in args.k_list if kk < N]
                refs = bp.references(q, kt, v, ks)
                mask = torch.zeros(N, N, device=device, dtype=dtype
                                   ).masked_fill_(eye, float("-inf"))
                bm_dense = create_block_mask(
                    lambda b, h, qi, ki: qi != ki, None, None, N, N,
                    device=device)
                dbest, dok, dall = bp.search(
                    bp.dense_space(N, d),
                    lambda c: bp.dense_call(q, kt, v, c, prec),
                    refs["dense"], tol, device, args.search_reps)
                dense_search.append({"mode": mode, "head_dim": d,
                                     "n_agents": N, "best": dbest,
                                     "admitted": dok, "space": dall})
                fns: dict[str, Callable[[], Any]] = {
                    "sdpa": lambda: bp.sdpa(q, kt, v, mask),
                    "flex_dense": lambda: bp.flex_dense(q, kt, v, bm_dense),
                    "dense_tuned": (lambda c=dbest["cfg"]:
                                    bp.dense_call(q, kt, v, c, prec)),
                }
                checks: dict[int, Any] = {}
                tiling_src: dict[int, str] = {}
                stream_cfg: dict[int, dict] = {}
                for kk in ks:
                    key = (mode, d, N, kk, B)
                    cfg, src = scfg.get(key), "per_format_search"
                    if cfg is None and key in ucfg:
                        # No tiling met the row criterion in this format (the
                        # format, not the tiling, changes the retained set):
                        # use the fastest tiling of a search that timed them
                        # all anyway. Flagged excluded below.
                        cfg, src = ucfg[key], "unadmitted_search"
                    if cfg is None:
                        cfg, src = fp32_cfg(d, N, kk, B), "fp32_search"
                    tiling_src[kk] = src
                    stream_cfg[kk] = cfg
                    fns[f"streaming@{kk}"] = (
                        lambda kk=kk, c=cfg: stream(q, kt, v, kk, c, prec))
                    if key in rcfg:
                        fns[f"fused_resident@{kk}"] = (
                            lambda kk=kk, c=rcfg[key]:
                            bp.sparse_call(q, kt, v, kk, c, prec))
                    chk = {"timed_inputs": set_check(q, kt, v, kk, cfg, prec,
                                                     tol), "extra_seeds": []}
                    for sd in range(args.extra_seeds):
                        g2 = torch.Generator(device=device).manual_seed(
                            50_000 + 31 * sd + N + d)
                        q2, k2, v2 = (torch.randn(B, N, d, device=device,
                                                  generator=g2).to(dtype)
                                      for _ in range(3))
                        chk["extra_seeds"].append(
                            set_check(q2, k2, v2, kk, cfg, prec, tol))
                        del q2, k2, v2
                    allc = [chk["timed_inputs"]] + chk["extra_seeds"]
                    chk["pooled_set_mismatch_frac"] = (
                        sum(c["set_mismatch_rows"] for c in allc)
                        / sum(c["rows"] for c in allc))
                    chk["excluded"] = (
                        chk["timed_inputs"]["set_mismatch_frac"] > bp.ROW_TOL
                        or src != "per_format_search")
                    checks[kk] = chk
                deviation = {}
                with torch.no_grad():
                    for n, f in fns.items():
                        ref = (refs["dense"] if "@" not in n
                               else refs[int(n.split("@")[1])])
                        deviation[n] = dict(zip(
                            ("max_abs", "rows_deviating_frac"),
                            bp.disagreement(f(), ref, tol)))
                torch.cuda.synchronize(device)
                for tmode in ("eager", "graph"):
                    timed = fns if tmode == "eager" else as_graphs(fns, device)
                    raw = time_group(timed, device, args.reps, args.warmup,
                                     args.runs)
                    for kk in checks:
                        sub = {n: raw[n] for n in DENSE}
                        sub.update({n.split("@")[0]: raw[n] for n in raw
                                    if n.endswith(f"@{kk}")})
                        g = summarise(sub, args.n_boot, rng)
                        g.update({
                            "format": mode, "mode": tmode, "n_agents": N,
                            "k": kk, "head_dim": d, "batch": B,
                            "streaming_cfg": stream_cfg[kk],
                            "streaming_tiling_source": tiling_src[kk],
                            "resident_cfg": rcfg.get((mode, d, N, kk, B)),
                            "dense_tuned_cfg": dbest["cfg"],
                            "retained_set": checks[kk],
                            "excluded": checks[kk]["excluded"],
                            "deviation": {n: deviation[n] for n in deviation
                                          if "@" not in n
                                          or n.endswith(f"@{kk}")},
                            "raw_ms": sub})
                        groups.append(g)
                        r = g["ratios"]
                        print(f"{mode} {tmode:<5} d={d:<2} N={N:<4} k={kk} "
                              f"best={g['best_dense']:<11} "
                              + " ".join(f"{n} {r[n]['ratio_of_minima']:.2f}/"
                                         f"{r[n]['paired_median']:.2f}"
                                         for n in ("streaming",
                                                   "fused_resident")
                                         if n in r)
                              + f" setmis {checks[kk]['timed_inputs']['set_mismatch_frac']:.2e}"
                              + (" EXCLUDED" if g["excluded"] else ""),
                              flush=True)
                    del timed, raw
                dump()
                del q, kt, v, refs, fns
                torch.cuda.empty_cache()
    bp.set_tf32(False)
    dump()
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
