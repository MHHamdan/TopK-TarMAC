"""Promote the adaptive gate's memory cost to a measured, first-class result.

Three things, in order:

1. **Derive** the activation-memory cost of each arm in closed form.
2. **Validate** the derivation against measured peak VRAM by fitting the
   empirical scaling exponent p in peak ~ N^p. A derivation nobody checked
   against a measurement is exactly the failure this project is documenting,
   so the exponent is fitted, not asserted.
3. **Locate the cliff**: the N at which each arm exhausts a given device.
   Reported for the full card and for the memory actually free on the shared
   host, because those are different numbers and only one of them is a
   property of the hardware.

Reads the microbenchmark JSONs; writes results/memory_analysis.json and a
Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Closed forms for the dominant activation tensor of each arm, in elements.
# B = batch, N = agents, k = kept peers, d = msg_dim.
DERIVATION = {
    "dense": {
        "formula": "B*N^2",
        "order": "O(B N^2)",
        "why": "the (B,N,N) softmax weight matrix is materialised and kept for "
               "the backward pass",
    },
    "dense_sdpa": {
        "formula": "B*N*d",
        "order": "O(B N d)",
        "why": "the fused kernel never materialises the weight matrix; only "
               "q,k,v and the output are resident",
    },
    "topk_masked": {
        "formula": "B*N^2",
        "order": "O(B N^2)",
        "why": "same weight matrix as dense, plus the top-k mask and the "
               "renormalised copy -- strictly more than dense, never less",
    },
    "topk_gather": {
        "formula": "B*N^2 + B*N*k*d",
        "order": "O(B N^2 + B N k d)",
        "why": "the score matrix is still needed to rank peers, and the "
               "gathered values add B*N*k*d. Gathering is only memory-"
               "favourable when k*d < N; at d=16 that needs k < N/16",
    },
    "adaptive": {
        "formula": "B*N*(N-1)*N",
        "order": "O(B N^3)",
        "why": "the gate broadcasts a (1,1,K,N) cumulative-keep tensor against "
               "(B,N,K,1) gate probabilities with K=N-1, materialising "
               "(B,N,N-1,N) before the sum over K (mappo.py, "
               "_forward adaptive_topk branch)",
    },
}


def elements(arm: str, b: int, n: int, k: int, d: int) -> float:
    if arm == "dense_sdpa":
        return b * n * d
    if arm in ("dense", "topk_masked"):
        return b * n * n
    if arm == "topk_gather":
        return b * n * n + b * n * k * d
    if arm == "adaptive":
        return b * n * max(n - 1, 1) * n
    raise ValueError(arm)


def fit_exponent(ns: list[int], vals: list[float]) -> float | None:
    """Least-squares slope of log(val) vs log(N) -- the empirical order."""
    pts = [(math.log(n), math.log(v)) for n, v in zip(ns, vals) if v and v > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den if den else None


def oom_n(arm: str, b: int, k_frac: float, d: int, capacity_mib: float,
          itemsize: int = 4, n_max: int = 100_000) -> int | None:
    """Smallest N whose dominant activation alone exceeds `capacity_mib`."""
    for n in range(2, n_max):
        k = max(1, int(k_frac * (n - 1)))
        mib = elements(arm, b, n, k, d) * itemsize / 2 ** 20
        if mib > capacity_mib:
            return n
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", default=[
        str(ROOT / "results" / "microbenchmark_comm.json")])
    p.add_argument("--card-mib", type=float, default=97887.0,
                   help="total VRAM of an unshared device")
    p.add_argument("--out", default=str(ROOT / "results" / "memory_analysis.json"))
    p.add_argument("--md", default=str(ROOT / "results" / "memory_analysis.md"))
    args = p.parse_args()

    payloads = [json.loads(Path(f).read_text()) for f in args.inputs
                if Path(f).exists()]
    if not payloads:
        print("no input microbenchmark JSON found; run the campaign first")
        return 1

    report: dict[str, Any] = {
        "derivation": DERIVATION,
        "sources": [Path(f).name for f in args.inputs if Path(f).exists()],
        "measured": {},
        "oom_thresholds": {},
    }

    for pay in payloads:
        b = pay["batch"]
        d = pay["msg_dim"]
        itemsize = {"fp32": 4, "bf16": 2, "fp16": 2}[pay.get("dtype", "fp32")]
        tag = f"batch{b}_{pay.get('dtype', 'fp32')}_{pay['device']}"
        free_mib = pay.get("device_free_mib_at_start")

        by_arm: dict[str, dict[str, list]] = {}
        for r in pay["rows"]:
            if r["exec_mode"] != "eager":
                continue
            a = by_arm.setdefault(r["arm"], {"n": [], "peak": [], "pred": [],
                                             "oom_n": []})
            if r["oom"]:
                a["oom_n"].append(r["n_agents"])
                continue
            if r.get("peak_vram_mib_forward"):
                a["n"].append(r["n_agents"])
                a["peak"].append(r["peak_vram_mib_forward"])
                a["pred"].append(
                    elements(r["arm"], b, r["n_agents"], r["k"], d)
                    * itemsize / 2 ** 20)

        meas: dict[str, Any] = {}
        for arm, a in by_arm.items():
            # Fit on N >= 48 only: below that a fixed ~19 MiB of parameters and
            # workspace dominates and would flatten the exponent toward zero.
            big = [(n, pk, pr) for n, pk, pr in zip(a["n"], a["peak"], a["pred"])
                   if n >= 48]
            meas[arm] = {
                "n": a["n"],
                "peak_vram_mib": [round(v, 1) for v in a["peak"]],
                "predicted_dominant_mib": [round(v, 1) for v in a["pred"]],
                "fitted_exponent_all_n": fit_exponent(a["n"], a["peak"]),
                "fitted_exponent_n_ge_48": fit_exponent(
                    [x[0] for x in big], [x[1] for x in big]),
                # The raw fit is biased low by a fixed ~20 MiB of parameters
                # and allocator workspace. Subtracting the smallest-N peak as
                # a constant offset removes most of that bias.
                "fitted_exponent_offset_corrected": (
                    fit_exponent([x[0] for x in big[1:]],
                                 [x[1] - big[0][1] for x in big[1:]])
                    if len(big) > 2 else None),
                # The sharper test: does the closed form predict the measured
                # peak at the largest N, where the dominant term dominates?
                "largest_n": big[-1][0] if big else None,
                "predicted_over_measured_at_largest_n": (
                    round(big[-1][2] / big[-1][1], 4) if big else None),
                "analytic_order": DERIVATION.get(arm, {}).get("order"),
                "oom_at_n": sorted(a["oom_n"]) or None,
            }
        report["measured"][tag] = {
            "batch": b, "msg_dim": d, "dtype": pay.get("dtype", "fp32"),
            "device": pay["device"], "device_name": pay.get("device_name"),
            "free_mib_at_start": free_mib,
            "arms": meas,
        }

        k_frac = (pay.get("k_frac") or [0.25])
        kf = k_frac[0] if isinstance(k_frac, list) else k_frac
        report["oom_thresholds"][tag] = {
            arm: {
                "n_oom_full_card": oom_n(arm, b, kf, d, args.card_mib, itemsize),
                "n_oom_free_on_shared_host": (
                    oom_n(arm, b, kf, d, free_mib, itemsize) if free_mib else None),
            }
            for arm in DERIVATION
        }

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    # ---- Markdown ------------------------------------------------------
    lines = ["# Activation memory of the communication arms", "",
             "Derived, then checked against measurement. Generated by "
             "`scripts/analyse_memory.py`.", "",
             "## 1. Derivation", "",
             "| Arm | Dominant activation | Order | Why |",
             "|---|---|---|---|"]
    for arm, v in DERIVATION.items():
        lines.append(f"| `{arm}` | `{v['formula']}` | {v['order']} | {v['why']} |")

    for tag, m in report["measured"].items():
        lines += ["", f"## 2. Measured — {tag}", "",
                  f"batch {m['batch']}, msg_dim {m['msg_dim']}, "
                  f"{m['dtype']}, {m['device_name']}", "",
                  "| Arm | Analytic order | Fitted N^p (offset-corrected) | "
                  "Peak VRAM at largest N (MiB) | Predicted/measured | OOM at N |",
                  "|---|---|---|---|---|---|"]
        for arm, a in m["arms"].items():
            exp = (a.get("fitted_exponent_offset_corrected")
                   or a["fitted_exponent_n_ge_48"])
            peak = a["peak_vram_mib"][-1] if a["peak_vram_mib"] else None
            ratio = a.get("predicted_over_measured_at_largest_n")
            lines.append(
                f"| `{arm}` | {a['analytic_order']} | "
                f"{exp:.2f} | {peak} | {ratio if ratio else '—'} | "
                f"{a['oom_at_n'][0] if a['oom_at_n'] else '—'} |"
                if exp is not None else
                f"| `{arm}` | {a['analytic_order']} | — | {peak} | "
                f"{ratio if ratio else '—'} | "
                f"{a['oom_at_n'][0] if a['oom_at_n'] else '—'} |")

    for tag, t in report["oom_thresholds"].items():
        lines += ["", f"## 3. Predicted OOM boundary — {tag}", "",
                  "Smallest N at which the dominant activation tensor alone "
                  "exceeds the budget.", "",
                  "| Arm | Full 97 887 MiB card | Free on the shared host |",
                  "|---|---|---|"]
        for arm, v in t.items():
            lines.append(f"| `{arm}` | {v['n_oom_full_card'] or '>1e5'} | "
                         f"{v['n_oom_free_on_shared_host'] or '>1e5'} |")

    Path(args.md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}\nwrote {args.md}")
    for tag, m in report["measured"].items():
        print(f"\n[{tag}]")
        for arm, a in m["arms"].items():
            e = (a.get("fitted_exponent_offset_corrected")
                 or a["fitted_exponent_n_ge_48"])
            ratio = a.get("predicted_over_measured_at_largest_n")
            print(f"  {arm:<13} analytic {str(a['analytic_order']):<18} "
                  f"fitted N^{e:.2f}   predicted/measured at N="
                  f"{a.get('largest_n')}: {ratio}"
                  if e is not None else
                  f"  {arm:<13} analytic {a['analytic_order']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
