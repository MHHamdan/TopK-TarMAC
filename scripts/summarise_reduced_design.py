"""Summarise the reduced-design training arms, tier by tier.

Reports mean +/- 95% bootstrap CI of final evaluation return per arm, and the
paired contrasts each tier exists to answer. The tier-1 contrast is a gate:
if dense attention-comm does not beat comm-free MAPPO on a task where the
channel is necessary, nothing measured in tiers 2-4 is interpretable
(D-011, D-027).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

# The six pre-registered contrasts, verbatim from results/PREREGISTRATION.md
# (committed before any tier-2/3/4 result was inspected). Order, direction and
# membership are fixed; nothing may be added, dropped or re-specified here.
CONTRASTS = [
    ("C1", 2, "Does WHICH peers attention selects carry information?",
     "red_po6_gather_k2", "red_po6_random_k2"),
    ("C2", 2, "Cost of sparsifying to k=2 against a working dense channel",
     "red_po6_gather_k2", "red_po6_dense"),
    ("C3", 3, "Is return monotone in k?",
     "red_po6_gather_k4", "red_po6_gather_k1"),
    ("C4", 3, "Sparsification cost at the largest trained N",
     "red_po12_gather_k3", "red_po12_dense"),
    ("C5", 4, "Does a Lagrangian budget recover dense return?",
     "red_po6_lagrangian", "red_po6_dense"),
    ("C6", 4, "Does entmax recover dense return?",
     "red_po6_entmax", "red_po6_dense"),
]
PRIMARY = "C1"

# The tier-1 gate. Inspected before the pre-registration was written, so it is
# reported for completeness and EXCLUDED from the multiplicity correction.
GATE = [
    ("G1", 1, "channel works on simple_reference", "red_ref_dense", "red_ref_mappo"),
    ("G2", 1, "channel works on PO simple_spread N=6", "red_po6_dense", "red_po6_mappo"),
]


def perm_test(a: list[float], b: list[float]) -> float:
    """Exact two-sided permutation p-value on the difference of means.

    With 5 vs 5 seeds there are C(10,5) = 252 assignments, so this enumerates
    rather than samples. The smallest attainable p is 2/252 = 0.0079, which is
    why the pre-registration fixed the power expectation in advance.
    """
    from itertools import combinations
    pool = a + b
    n = len(a)
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    idx = range(len(pool))
    count = tot = 0
    for pick in combinations(idx, n):
        left = [pool[i] for i in pick]
        right = [pool[i] for i in idx if i not in set(pick)]
        tot += 1
        if abs(statistics.fmean(left) - statistics.fmean(right)) >= obs - 1e-12:
            count += 1
    return count / tot


def hodges_lehmann(a: list[float], b: list[float]) -> float:
    """Median of all pairwise differences -- distribution-free effect size."""
    return statistics.median([x - y for x in a for y in b])


def cohens_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled if pooled else 0.0


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def final_returns(arm: str) -> list[float]:
    """Final eval return per seed, preferring final_eval.json over the CSV."""
    out = []
    for seed_dir in sorted((R / arm).glob("seed_*")) if (R / arm).exists() else []:
        fe = seed_dir / "final_eval.json"
        if fe.exists():
            d = json.loads(fe.read_text())
            v = d.get("eval_return_mean", d.get("return_mean"))
            if v is not None:
                out.append(float(v))
                continue
        csv_p = seed_dir / "metrics.csv"
        if csv_p.exists():
            rows = [r for r in csv.DictReader(csv_p.open())
                    if r.get("eval_return_mean")]
            if rows:
                out.append(float(rows[-1]["eval_return_mean"]))
    return out


def boot_ci(vals: list[float], reps: int = 10_000,
            seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(vals, k=len(vals)))
                   for _ in range(reps))
    return (means[int(0.025 * reps)], means[int(0.975 * reps)])


def mech(arm: str) -> dict[str, float]:
    """Mean of the mechanism columns over the last decile of training."""
    vals: dict[str, list[float]] = {}
    for seed_dir in sorted((R / arm).glob("seed_*")) if (R / arm).exists() else []:
        csv_p = seed_dir / "metrics.csv"
        if not csv_p.exists():
            continue
        rows = list(csv.DictReader(csv_p.open()))
        for row in rows[max(0, int(len(rows) * 0.9)):]:
            for key in ("mean_k", "attn_entropy", "lagrange_lambda"):
                if row.get(key):
                    vals.setdefault(key, []).append(float(row[key]))
    return {k: statistics.fmean(v) for k, v in vals.items() if v}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(R / "reduced_design_summary.md"))
    args = p.parse_args()

    manifest = json.loads((ROOT / "configs" / "reduced" / "MANIFEST.json").read_text())
    lines = ["# Reduced design — results", "",
             "Generated by `scripts/summarise_reduced_design.py`. "
             "CIs are 95% bootstrap over seeds.", "",
             "| Tier | Arm | seeds | final return | 95% CI | mean_k | "
             "attn entropy | lambda |", "|---|---|---|---|---|---|---|---|"]

    data: dict[str, list[float]] = {}
    for a in manifest["arms"]:
        arm = a["run_name"]
        vals = final_returns(arm)
        data[arm] = vals
        m = mech(arm)
        if not vals:
            lines.append(f"| {a['tier']} | `{arm}` | 0 | _not run_ | — | — | — | — |")
            continue
        lo, hi = boot_ci(vals)
        lines.append(
            f"| {a['tier']} | `{arm}` | {len(vals)} | "
            f"{statistics.fmean(vals):.2f} | [{lo:.2f}, {hi:.2f}] | "
            f"{m.get('mean_k', float('nan')):.2f} | "
            f"{m.get('attn_entropy', float('nan')):.2f} | "
            f"{m.get('lagrange_lambda', float('nan')):.2f} |")

    def analyse(cid, tier, q, a_name, b_name):
        a_vals, b_vals = data.get(a_name, []), data.get(b_name, [])
        if len(a_vals) < 2 or len(b_vals) < 2:
            return None
        rng = random.Random(0)
        diffs = sorted(
            statistics.fmean(rng.choices(a_vals, k=len(a_vals)))
            - statistics.fmean(rng.choices(b_vals, k=len(b_vals)))
            for _ in range(10_000))
        return {
            "id": cid, "tier": tier, "q": q, "a": a_name, "b": b_name,
            "mean_diff": statistics.fmean(a_vals) - statistics.fmean(b_vals),
            "hl": hodges_lehmann(a_vals, b_vals),
            "d": cohens_d(a_vals, b_vals),
            "ci": (diffs[250], diffs[9750]),
            "p": perm_test(a_vals, b_vals),
        }

    fam = [r for r in (analyse(*c) for c in CONTRASTS) if r]
    adj = holm([r["p"] for r in fam])
    for r, pa in zip(fam, adj):
        r["p_holm"] = pa

    lines += ["", "## Pre-registered contrasts (tiers 2-4)", "",
              "Analysis fixed in `results/PREREGISTRATION.md` before any of "
              "these results was inspected. Exact permutation test (252 "
              "arrangements), Holm-Bonferroni across the six. "
              f"**{PRIMARY} is the primary contrast.**", "",
              "| ID | Question | A | B | A-B | Hodges-Lehmann | Cohen's d | "
              "95% CI | p (exact) | p (Holm) | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in fam:
        sig = r["p_holm"] < 0.05
        verdict = (("**A > B**" if r["mean_diff"] > 0 else "**B > A**") if sig
                   else "no separation (under-powered)")
        star = " ⭐" if r["id"] == PRIMARY else ""
        lines.append(
            f"| {r['id']}{star} | {r['q']} | `{r['a']}` | `{r['b']}` | "
            f"{r['mean_diff']:+.2f} | {r['hl']:+.2f} | {r['d']:+.2f} | "
            f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['p']:.4f} | "
            f"{r['p_holm']:.4f} | {verdict} |")

    lines += ["", "Smallest attainable exact p at 5v5 is 2/252 = 0.0079; "
              "Holm's strictest threshold for six tests is 0.0083. A "
              "non-significant row here is **under-powered, not evidence of "
              "absence**, and no equivalence claim is available because no "
              "TOST margin was pre-registered.", ""]

    gate = [r for r in (analyse(*c) for c in GATE) if r]
    lines += ["## Tier-1 gate (inspected before pre-registration; excluded "
              "from the correction above)", "",
              "| ID | Question | A-B | 95% CI | p (exact) | verdict |",
              "|---|---|---|---|---|---|"]
    for r in gate:
        lines.append(f"| {r['id']} | {r['q']} | {r['mean_diff']:+.2f} | "
                     f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | "
                     f"{r['p']:.4f} | "
                     f"{'**A > B**' if r['ci'][0] > 0 else 'no separation'} |")
    lines += [""]

    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
