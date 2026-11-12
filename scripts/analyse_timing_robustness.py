"""Derive fixed-baseline paired summaries; never rewrite measurement JSON."""
from __future__ import annotations

import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent.parent


def paired_summary(raw: dict[str, list[list[float]]]) -> dict:
    flat = {n: [v for run in runs for v in run] for n, runs in raw.items()}
    dense = sorted(n for n in flat if n.startswith("dense_") or n in {"sdpa", "flex_dense"})
    if not dense:
        raise ValueError("no dense baseline")
    # One baseline for the whole configuration, chosen by sample median.
    # Selection uses the same observations, not a held-out selection phase.
    baseline = min(dense, key=lambda n: st.median(flat[n]))
    denom = flat[baseline]
    if any(len(x) != len(denom) for x in flat.values()):
        raise ValueError("unpaired sample lengths")
    layout = [len(run) for run in raw[baseline]]
    if any([len(run) for run in runs] != layout for runs in raw.values()):
        raise ValueError("unpaired run layout")
    oracle = [min(flat[n][i] for n in dense) for i in range(len(denom))]
    out = {}
    for name, x in flat.items():
        if name in dense:
            continue
        pairs = [a / b for a, b in zip(x, denom)]
        by_run = [st.median(a / b for a, b in zip(xs, ds))
                  for xs, ds in zip(raw[name], raw[baseline])]
        out[name] = {"fixed_baseline_paired_median": st.median(pairs),
                     "pointwise_best_dense_paired_median": st.median(a / b for a, b in zip(x, oracle)),
                     "fixed_baseline_per_run_medians": by_run,
                     "ratio_of_medians_fixed_baseline": st.median(x) / st.median(denom),
                     "ratio_of_minima_best_dense": min(x) / min(min(flat[n]) for n in dense)}
    return {"fixed_baseline": baseline, "n_repetitions": len(denom), "ratios": out}


def derive() -> dict:
    rows = []
    for folder in [ROOT / "results", *sorted((ROOT / "results" / "portability").glob("*"))]:
        for name in ["fixed_set.json", "streaming_topk.json", "streaming_precision.json"]:
            path = folder / name
            if not path.exists():
                continue
            for group in json.loads(path.read_text()).get("groups", []):
                if group.get("excluded", False):
                    continue
                row = {"source": str(path.relative_to(ROOT)),
                       **{k: group.get(k) for k in ["mode", "format", "batch", "n_agents", "head_dim", "k"]},
                       **paired_summary(group["raw_ms"])}
                stats = group.get("stats", {})
                if "fixed_set" in stats:
                    row["residuals"] = {
                        n: {"ratio_of_minima": 1 - stats["fixed_set"]["min_ms"] / stats[n]["min_ms"],
                            "ratio_of_medians": 1 - stats["fixed_set"]["median_ms"] / stats[n]["median_ms"]}
                        for n in ["fused_stsel", "streaming"] if n in stats}
                rows.append(row)
    prescore = []
    for path in [*sorted((ROOT / "results").glob("prescore*.json")),
                 *sorted((ROOT / "results" / "portability").glob("*/prescore.json"))]:
        for r in json.loads(path.read_text())["rows"]:
            if r["n_agents"] != 512 or r["k"] != 4 or r["selector"] != "morton_window":
                continue
            prescore.append({"source": str(path.relative_to(ROOT)), "raw_pairings_available": False,
                             "n_agents": 512, "k": 4,
                             "ratio_of_minima": r["t_total_fused_ms"] / r["t_dense_best_ms"],
                             "ratio_of_medians": r["dispersion"]["total_fused"]["median_ms"] /
                                 min(x["median_ms"] for x in r["dense_dispersion"].values())})
    return {"analysis_version": "b1_fixed_baseline_v1",
            "scope": "Within existing sessions; dense baseline selected once by sample median. No held-out selection or new CI. Ratios of medians are not paired medians.",
            "groups": rows, "prescore_crossing": prescore}


if __name__ == "__main__":
    out = ROOT / "results" / "analysis" / "timing_robustness_b1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(derive(), indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
