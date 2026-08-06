"""Reduce the microbenchmark campaign to the tables the claim turns on.

The manuscript's efficiency claim is about a *gathered* top-k implementation.
The original benchmark compared dense against a *masked* one that executes the
same dense aggregation matmul, so it could only ever have shown top-k losing
(D-019). Every table here therefore reports three curves separately --
`dense`, `topk_masked`, `topk_gather` -- plus the fused `dense_sdpa` baseline,
because on this hardware that is what "dense attention" actually costs.

Latency is quoted as `min_ms`: the device is shared and contention can only
add time, so the minimum over the samples is the closest available estimate of
uncontended latency. Arms were timed interleaved rep-by-rep, so ratios between
arms within one N are paired and robust even where absolute values drift.

Writes results/efficiency_summary.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

ORDER = ["dense_sdpa", "dense", "topk_masked", "topk_gather", "adaptive"]
LABEL = {
    "dense_sdpa": "dense (fused SDPA)",
    "dense": "dense (unfused)",
    "topk_masked": "top-k masked",
    "topk_gather": "top-k gathered",
    "adaptive": "adaptive gate",
}


def load(name: str) -> dict[str, Any] | None:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def lat(row: dict, key: str = "forward") -> float | None:
    v = row.get(key)
    return v["min_ms"] if v else None


def rows_for(pay: dict, exec_mode: str = "eager") -> list[dict]:
    return [r for r in pay["rows"] if r.get("exec_mode") == exec_mode]


def headline(pay: dict, title: str) -> list[str]:
    """Three curves plus fused baseline, per N, with executed FLOPs."""
    out = [f"### {title}", "",
           f"batch {pay['batch']}, msg_dim {pay['msg_dim']}, "
           f"dtype {pay.get('dtype', 'fp32')}, k = 25% of peers, "
           f"{pay['device_name']}", "",
           "| N | k | " + " | ".join(f"{LABEL[a]} (ms)" for a in ORDER)
           + " | gathered/dense | gathered/SDPA | gathered MFLOP vs dense |",
           "|" + "---|" * (4 + len(ORDER))]
    ns = sorted({r["n_agents"] for r in rows_for(pay)})
    for n in ns:
        cells, by = [], {}
        for r in rows_for(pay):
            if r["n_agents"] == n:
                by[r["arm"]] = r
        k = by.get("topk_gather", {}).get("k", "—")
        for a in ORDER:
            r = by.get(a)
            if r is None:
                cells.append("—")
            elif r["oom"]:
                cells.append("**OOM**")
            else:
                cells.append(f"{lat(r):.4f}")
        g, d, s = by.get("topk_gather"), by.get("dense"), by.get("dense_sdpa")
        gd = (f"{lat(g) / lat(d):.2f}x" if g and d and not g["oom"]
              and not d["oom"] else "—")
        gs = (f"{lat(g) / lat(s):.2f}x" if g and s and not g["oom"]
              and not s["oom"] else "—")
        fl = (f"{g['flops_per_forward'] / 1e6:.0f} vs "
              f"{d['flops_per_forward'] / 1e6:.0f}" if g and d else "—")
        out.append(f"| {n} | {k} | " + " | ".join(cells)
                   + f" | {gd} | {gs} | {fl} |")
    return out + [""]


def crossover(pay: dict) -> list[str]:
    """Best case for the gather path across the whole (N, k) grid."""
    out = ["### Sparsity sweep — is there ANY k at which gathering wins?", "",
           f"batch {pay['batch']}, dtype {pay.get('dtype', 'fp32')}. For each "
           "N, the k that minimises gathered latency, and how that best case "
           "compares to dense.", "",
           "| N | best k | gathered (ms) | dense (ms) | fused SDPA (ms) | "
           "best gathered/dense | best gathered/SDPA | verdict |",
           "|---|---|---|---|---|---|---|---|"]
    ns = sorted({r["n_agents"] for r in rows_for(pay)})
    for n in ns:
        rs = [r for r in rows_for(pay) if r["n_agents"] == n and not r["oom"]]
        gs = [r for r in rs if r["arm"] == "topk_gather"]
        d = next((r for r in rs if r["arm"] == "dense"), None)
        s = next((r for r in rs if r["arm"] == "dense_sdpa"), None)
        if not gs or not d:
            continue
        best = min(gs, key=lambda r: lat(r))
        rd = lat(best) / lat(d)
        rs_ = lat(best) / lat(s) if s else None
        verdict = "gather wins" if rd < 1.0 else "dense wins"
        out.append(
            f"| {n} | {best['k']} | {lat(best):.4f} | {lat(d):.4f} | "
            f"{lat(s):.4f} | {rd:.2f}x | "
            f"{f'{rs_:.2f}x' if rs_ else '—'} | {verdict} |")
    return out + [""]


def exec_modes(pay: dict) -> list[str]:
    out = ["### Execution modes — does compilation or graph capture change it?",
           "",
           "CUDA-graph mode reports forward only: graph replays share a memory "
           "pool across arms, so an interleaved backward would read "
           "activations a later replay has overwritten.", "",
           "| N | arm | eager (ms) | torch.compile (ms) | CUDA graphs (ms) | "
           "best mode |", "|---|---|---|---|---|---|"]
    ns = sorted({r["n_agents"] for r in pay["rows"]})
    for n in ns:
        for a in ORDER:
            got = {}
            for m in ("eager", "compile", "cudagraph"):
                r = next((x for x in pay["rows"] if x["n_agents"] == n
                          and x["arm"] == a and x["exec_mode"] == m), None)
                if r and not r["oom"] and lat(r):
                    got[m] = lat(r)
            if not got:
                continue
            best = min(got, key=lambda m: got[m])
            out.append(
                f"| {n} | {LABEL[a]} | "
                + " | ".join(f"{got[m]:.4f}" if m in got else "—"
                             for m in ("eager", "compile", "cudagraph"))
                + f" | {best} |")
    return out + [""]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(R / "efficiency_summary.md"))
    args = p.parse_args()

    main_pay = load("microbenchmark_comm.json")
    if main_pay is None:
        print("run scripts/run_microbenchmarks.sh first")
        return 1

    lines = ["# Efficiency claim — measured", "",
             "Generated by `scripts/summarise_microbenchmarks.py`. Latency is "
             "`min_ms` over interleaved, paired samples; see the module "
             "docstring for the protocol and its limits.", ""]

    lines += ["## 1. Headline: three implementations of the same function", ""]
    lines += headline(main_pay, "fp32, batch 256")

    b4 = load("microbenchmark_comm_batch4096.json")
    if b4:
        lines += ["## 2. FLOP-bound regime", ""]
        lines += headline(b4, "fp32, batch 4096")

    bf = load("microbenchmark_bf16.json")
    if bf:
        lines += ["## 3. bf16", "",
                  "The dtype where the flash-attention kernels are reachable; "
                  f"backends available here: "
                  f"{bf.get('sdpa_backends', {}).get('available')}.", ""]
        lines += headline(bf, "bf16, batch 256")

    ks = load("microbenchmark_ksweep.json")
    if ks:
        lines += ["## 4. Is there a crossover in k?", ""]
        lines += crossover(ks)

    em = load("microbenchmark_execmodes.json")
    if em:
        lines += ["## 5. Execution modes", ""]
        lines += exec_modes(em)

    cpu = load("microbenchmark_cpu.json")
    if cpu:
        lines += ["## 6. CPU — is the result a GPU-kernel artifact?", "",
                  "If the ordering survives on a completely different "
                  "architecture, it is a property of the arithmetic and the "
                  "memory traffic, not of one vendor's kernels.", ""]
        lines += headline(cpu, "fp32, batch 256, CPU")

    # Provenance the reader needs to judge the numbers.
    lines += ["## Measurement conditions", "",
              "| Field | Value |", "|---|---|"]
    for key in ("device", "device_name", "capability", "torch", "cuda",
                "reps", "warmup", "runs", "timing", "collect_info_disabled",
                "clocks_locked", "clocks_at_start", "sdpa_backends",
                "device_free_mib_at_start", "device_total_mib"):
        if key in main_pay:
            lines.append(f"| `{key}` | {main_pay[key]} |")
    lines += ["",
              "GPU clocks could **not** be locked (`nvidia-smi -lgc` requires "
              "privileges this account lacks), so per-arm inter-run spread is "
              "recorded in the JSON for every cell and arms were interleaved "
              "rep-by-rep to make the comparison paired.", "",
              "**Not done:** replication on a second GPU generation. The "
              "2080 Ti host that produced the historical runs is no longer "
              "available (HARDWARE.md), and both devices here are the same "
              "Blackwell part, so cross-generation agreement is untested. "
              "The CPU section is the only cross-architecture evidence.", ""]

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
