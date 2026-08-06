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
    p.add_argument("--out", default=str(R / "drift_report.json"))
    p.add_argument("--md", default=str(R / "drift_report.md"))
    args = p.parse_args()

    base = json.loads(Path(args.base).read_text())
    comps = []
    for rp in args.repeats:
        if not Path(rp).exists():
            continue
        rep = json.loads(Path(rp).read_text())
        same = rep["device"] == base["device"]
        comps.append(compare(base, rep, "same device, ~21 h later, under "
                             "concurrent training load" if same else
                             "second device (same model), idle"))
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

    lines += ["", "## Reading", "",
              "Absolute latency is the less stable quantity, which is expected "
              "on a shared host with unlocked clocks. The arm *ratios* are what "
              "every claim in this paper is expressed in, and they are the "
              "quantity the interleaved protocol was designed to protect. "
              "Where the ratio agreement is tight, the conclusions carry across "
              "runs, devices and load; where it is not, the affected cells are "
              "visible above and should be read with that spread in mind.", ""]
    Path(args.md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"\nwrote {args.out}\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
