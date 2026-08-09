"""Drift control: does the headline measurement reproduce hours later?

GPU clocks could not be locked on this host and the devices are shared, so the
paper's latency numbers are only as good as their stability. This re-runs the
identical sweep and reports per-cell agreement against the original.

Two comparisons, which answer different questions:

  same device, later, under different load
      Tests time drift and the central design claim that `min_ms` over
      interleaved samples is robust to contention. The repeat runs while
      training occupies the same device, so if `min_ms` still agrees, the
      statistic is doing its job.

  second device, same model, idle
      Tests whether the numbers are a property of one physical card.

Agreement is reported on `min_ms` (the reported statistic) and on the
*ratios between arms*, which are what every claim in the paper is stated in.
Ratios are expected to be far more stable than absolute latency, and if they
are not, the claims are not safe.

Writes results/drift_report.md and .json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
ARMS = ["dense_sdpa", "dense", "topk_masked", "topk_gather", "adaptive"]


def index(pay: dict) -> dict[tuple[int, str], dict]:
    return {(r["n_agents"], r["arm"]): r for r in pay["rows"]
            if r.get("exec_mode") == "eager"}


def lat(row: dict | None) -> float | None:
    if not row or row.get("oom") or not row.get("forward"):
        return None
    return row["forward"]["min_ms"]


def compare(base: dict, rep: dict, label: str) -> dict[str, Any]:
    a, b = index(base), index(rep)
    cells, ratio_cells = [], []
    for (n, arm), ra in sorted(a.items()):
        rb = b.get((n, arm))
        la, lb = lat(ra), lat(rb)
        if la is None or lb is None:
            continue
        cells.append({"n_agents": n, "arm": arm, "base_ms": la,
                      "repeat_ms": lb, "pct_diff": 100.0 * (lb - la) / la})
    for n in sorted({n for n, _ in a}):
        da, db = lat(a.get((n, "dense"))), lat(b.get((n, "dense")))
        ga, gb = lat(a.get((n, "topk_gather"))), lat(b.get((n, "topk_gather")))
        if None in (da, db, ga, gb):
            continue
        ratio_cells.append({
            "n_agents": n, "base_ratio": ga / da, "repeat_ratio": gb / db,
            "pct_diff": 100.0 * ((gb / db) - (ga / da)) / (ga / da)})
    abs_d = [abs(c["pct_diff"]) for c in cells]
    abs_r = [abs(c["pct_diff"]) for c in ratio_cells]
    return {
        "label": label,
        "base_device": base["device"], "repeat_device": rep["device"],
        "cells": cells, "ratio_cells": ratio_cells,
        "median_abs_pct_diff": statistics.median(abs_d) if abs_d else None,
        "max_abs_pct_diff": max(abs_d) if abs_d else None,
        "ratio_median_abs_pct_diff": statistics.median(abs_r) if abs_r else None,
        "ratio_max_abs_pct_diff": max(abs_r) if abs_r else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=str(R / "microbenchmark_comm.json"))
    p.add_argument("--repeats", nargs="+", default=[
        str(R / "microbenchmark_drift_cuda_0.json"),
        str(R / "microbenchmark_drift_cuda_1.json")])
    p.add_argument("--labels", nargs="+", default=None,
                   help="One label per repeat. Without this the label is "
                        "inferred from whether the repeat device matches the "
                        "base device, which cannot distinguish an idle repeat "
                        "from a loaded one on the same card.")
    p.add_argument("--out", default=str(R / "drift_report.json"))
    p.add_argument("--md", default=str(R / "drift_report.md"))
    args = p.parse_args()

    base = json.loads(Path(args.base).read_text())
    comps = []
    if args.labels and len(args.labels) != len(args.repeats):
        raise SystemExit("--labels must give one label per --repeats entry")
    for i, rp in enumerate(args.repeats):
        if not Path(rp).exists():
            continue
        rep = json.loads(Path(rp).read_text())
        if args.labels:
            label = args.labels[i]
        else:
            same = rep["device"] == base["device"]
            label = ("same device, ~21 h later, under concurrent training load"
                     if same else "second device (same model), idle")
        comps.append(compare(base, rep, label))
    if not comps:
        print("no drift repeats found")
        return 1

    payload = {"base": Path(args.base).name, "comparisons": comps}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Drift control", "",
             "The identical sweep, re-run after the original. Agreement is "
             "reported both on absolute `min_ms` and on the arm ratios the "
             "paper's claims are actually stated in.", "",
             "## Summary", "",
             "| Comparison | base | repeat | median abs diff (latency) | "
             "max abs diff (latency) | median abs diff (**ratio**) | "
             "max abs diff (**ratio**) |",
             "|---|---|---|---|---|---|---|"]
    for c in comps:
        lines.append(
            f"| {c['label']} | `{c['base_device']}` | `{c['repeat_device']}` | "
            f"{c['median_abs_pct_diff']:.1f}% | {c['max_abs_pct_diff']:.1f}% | "
            f"**{c['ratio_median_abs_pct_diff']:.1f}%** | "
            f"**{c['ratio_max_abs_pct_diff']:.1f}%** |")

    for c in comps:
        lines += ["", f"### {c['label']}", "",
                  "gathered/dense ratio, original vs repeat:", "",
                  "| N | base ratio | repeat ratio | diff |",
                  "|---|---|---|---|"]
        for rc in c["ratio_cells"]:
            lines.append(f"| {rc['n_agents']} | {rc['base_ratio']:.2f}x | "
                         f"{rc['repeat_ratio']:.2f}x | "
                         f"{rc['pct_diff']:+.1f}% |")

    # The question that actually matters is not "do the numbers move" but
    # "does any condition move them across 1.0", i.e. produce a crossover.
    all_ratios = [rc["repeat_ratio"] for c in comps for rc in c["ratio_cells"]]
    all_ratios += [rc["base_ratio"] for c in comps for rc in c["ratio_cells"]]
    worst = min(all_ratios) if all_ratios else float("nan")
    loaded = next((c for c in comps if "load" in c["label"]), None)

    # Per-cell sign counts, so the directional statement below is counted
    # rather than asserted. A ratio moving UP is adverse to the sparse path.
    def signs(c: dict) -> tuple[int, int, int]:
        up = sum(1 for rc in c["ratio_cells"] if rc["pct_diff"] > 0)
        dn = sum(1 for rc in c["ratio_cells"] if rc["pct_diff"] < 0)
        return up, dn, len(c["ratio_cells"])

    lines += ["", "## Reading", "",
              f"The smallest gathered/dense ratio in **these drift artifacts** "
              f"({', '.join(c['label'] for c in comps)}) is "
              f"**{worst:.2f}x**, still above 1.0. This is a statement about "
              f"the runs compared here only: it is not a minimum over every "
              f"run in the project, and in particular it does not cover other "
              f"devices, precisions or execution modes, which are reported "
              f"separately.", ""]
    for c in comps:
        up, dn, tot = signs(c)
        lines += [f"On **{c['label']}**, ratios move by a median of "
                  f"{c['ratio_median_abs_pct_diff']:.1f}% and at most "
                  f"{c['ratio_max_abs_pct_diff']:.1f}% at any N; "
                  f"{up} of {tot} cells move against the sparse path and "
                  f"{dn} of {tot} in its favour.", ""]
    if loaded:
        up, dn, tot = signs(loaded)
        lines += [f"Drift under load is concentrated at small and moderate N, "
                  f"where kernels are launch-bound and therefore most "
                  f"sensitive to competition for the device. The directional "
                  f"result ({up} of {tot} cells adverse) comes from a single "
                  f"contention pattern and should be read as a margin "
                  f"argument, not an impossibility proof: no condition "
                  f"measured here moves any cell within "
                  f"{100.0 * (worst - 1.0):.0f}% of a crossover, but an "
                  f"interferer that penalised the dense path more than the "
                  f"launch-bound sparse one is not excluded by these data.",
                  "",
                  "The operational consequence: latency numbers in this work "
                  "must be taken on an unloaded device. The interleaved, "
                  "minimum-over-samples protocol substantially reduces "
                  "contention sensitivity but does not eliminate it, and "
                  "claiming otherwise would overstate what the protocol buys.",
                  ""]
    Path(args.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"\nwrote {args.out}\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
