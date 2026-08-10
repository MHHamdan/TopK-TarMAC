"""Verify that the numbers in the manuscript match the measurement artifacts.

This paper's central finding is that a reported quantity drifted away from the
quantity the code actually computed. It would be indefensible to let the same
thing happen between our own results and our own prose, so every headline
number in the LaTeX is checked here against the JSON it came from. Run it
before any commit that touches paper/ or results/.

Exit code 1 on any mismatch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
PAPER = ROOT / "paper" / "sections"

TOL = 0.005  # 0.5% -- the tables are quoted to 3 decimal places


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def row(pay: dict, n: int, arm: str, exec_mode: str = "eager") -> dict | None:
    for r in pay["rows"]:
        if (r["n_agents"] == n and r["arm"] == arm
                and r.get("exec_mode") == exec_mode):
            return r
    return None


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def main() -> int:
    fails: list[str] = []
    checks = 0

    # ---- Table: three curves (04_measurement.tex) -----------------------
    pay = load("microbenchmark_comm.json")
    tex = (PAPER / "04_measurement.tex").read_text()
    arms = {"fused": "dense_sdpa", "unfused": "dense",
            "masked": "topk_masked", "gathered": "topk_gather"}
    # Rows look like:  6   & 1   & 0.117 & 0.124 & 0.174 & 0.146 & 1.17 & ...
    pattern = re.compile(
        r"^(\d+)\s*&\s*(\d+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)"
        r"\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&", re.M)
    found = 0
    for m in pattern.finditer(tex):
        n = int(m.group(1))
        quoted = {
            "fused": float(m.group(3)), "unfused": float(m.group(4)),
            "masked": float(m.group(5)), "gathered": float(m.group(6)),
        }
        r_gd, r_gf = float(m.group(7)), float(m.group(8))
        if row(pay, n, "dense") is None:
            continue
        found += 1
        for label, arm in arms.items():
            r = row(pay, n, arm)
            if r is None or r["oom"]:
                continue
            actual = r["forward"]["min_ms"]
            checks += 1
            if abs(actual - quoted[label]) > 0.0006:  # quoted to 3 dp
                fails.append(f"headline N={n} {label}: paper {quoted[label]} "
                             f"vs artifact {actual:.4f}")
        g = row(pay, n, "topk_gather")["forward"]["min_ms"]
        d = row(pay, n, "dense")["forward"]["min_ms"]
        f = row(pay, n, "dense_sdpa")["forward"]["min_ms"]
        checks += 2
        if not close(g / d, r_gd, 0.01):
            fails.append(f"headline N={n} ratio/dense: paper {r_gd} "
                         f"vs artifact {g / d:.3f}")
        if not close(g / f, r_gf, 0.01):
            fails.append(f"headline N={n} ratio/fused: paper {r_gf} "
                         f"vs artifact {g / f:.3f}")
    if found < 8:
        fails.append(f"headline table: matched {found} rows, expected 8")

    # ---- Crossover surface count ---------------------------------------
    cs = load("crossover_surface.json")
    live = [c for c in cs["cells"] if not c.get("skipped")]
    wins_d = sum(1 for c in live if c["wins_vs_dense"])
    wins_f = sum(1 for c in live if c["wins_vs_sdpa"])
    tex_all = "\n".join(p.read_text() for p in PAPER.glob("*.tex"))
    checks += 3
    if f"$0$ of ${len(live)}$" not in tex_all:
        fails.append(f"crossover: artifact says {wins_d} of {len(live)} wins "
                     f"vs dense; paper text does not state '$0$ of ${len(live)}$'")
    if wins_d != 0:
        fails.append(f"crossover: artifact now has {wins_d} wins vs dense, "
                     f"paper claims 0")
    if wins_f != 1:
        fails.append(f"crossover: artifact has {wins_f} wins vs fused, "
                     f"paper claims 1")

    # ---- Component decomposition ---------------------------------------
    cd = load("component_decomposition.json")
    big = max(cd["rows"], key=lambda r: r["n_agents"])
    st = big["stages"]
    ratio_sel = st["topk_select"]["min_ms"] / st["agg_dense"]["min_ms"]
    ratio_agg = st["agg_sparse"]["min_ms"] / st["agg_dense"]["min_ms"]
    checks += 3
    if not close(ratio_sel, 13.7, 0.02):
        fails.append(f"decomp: selection/agg_dense is {ratio_sel:.1f}x, "
                     f"paper says 13.7x")
    if not close(ratio_agg, 3.4, 0.03):
        fails.append(f"decomp: agg_sparse/agg_dense is {ratio_agg:.1f}x, "
                     f"paper says 3.4x")
    if big["aggregation_saving_ms"] >= 0:
        fails.append("decomp: aggregation saving is no longer negative; the "
                     "paper's central mechanistic claim has changed")

    # ---- Roofline -------------------------------------------------------
    rf = load("roofline.json")
    checks += 2
    for dev, m in rf["machines"].items():
        ridge = m["machine_balance_flops_per_byte"]
        want = 52.1 if dev.startswith("cuda") else 1.35
        if not close(ridge, want, 0.02):
            fails.append(f"roofline {dev}: ridge point {ridge:.2f}, "
                         f"paper says {want}")
    big_r = max(rf["rows"], key=lambda r: r["n_agents"])
    checks += 1
    if not close(big_r["gather"]["ai"], 0.66, 0.02):
        fails.append(f"roofline: gather AI {big_r['gather']['ai']:.2f}, "
                     f"paper says 0.66")

    # The manuscript now scopes the CPU claim per ridge estimate rather than
    # asserting it under all of them: dense clears the low 1.35 estimate at
    # every N, and the high 3.68 estimate only from N=192. Both halves are
    # checked, because the earlier unscoped version of this sentence was
    # false against the high estimate.
    lo, hi = 1.35, 3.68
    dense_ai = {r["n_agents"]: r["dense"]["ai"] for r in rf["rows"]}
    checks += 2
    if not all(ai > lo for ai in dense_ai.values()):
        fails.append(f"roofline: dense AI does not clear the {lo} CPU ridge "
                     f"at every N: {dense_ai}")
    above_hi = {n for n, ai in dense_ai.items() if ai > hi}
    if above_hi != {n for n in dense_ai if n >= 192}:
        fails.append(f"roofline: dense AI clears the {hi} CPU ridge at "
                     f"{sorted(above_hi)}, paper says N >= 192 only")

    # ---- SDPA backend probe --------------------------------------------
    # The paper claims (a) FlashAttention serves no arm under any mask form,
    # (b) the fused arm runs the CUTLASS memory-efficient kernel, and (c) a
    # FlashAttention-served baseline would be faster, so the reported fused
    # gap is a lower bound. All three are checked against the recorded log.
    probe_p = R / "sdpa_backend_probe.json"
    if probe_p.exists():
        pb = json.loads(probe_p.read_text())
        checks += 3
        masked = [p for p in pb["probes"] for m, avail in p["by_mask"].items()
                  if m != "no_mask" and "flash" in avail]
        if masked:
            fails.append("sdpa probe: FlashAttention is available under a "
                         "mask, contradicting the manuscript")
        disp = pb["dispatched_kernel_with_additive_mask"]
        if not all("cutlass" in disp[dt].lower() for dt in ("fp32", "bf16")):
            fails.append(f"sdpa probe: fused arm is not served by CUTLASS: "
                         f"{disp}")
        lat = pb["unmasked_bf16_backend_latency"]
        for n in (192, 512):
            if lat[f"N{n}_flash_min_ms"] >= lat[f"N{n}_mem_efficient_min_ms"]:
                fails.append(f"sdpa probe: at N={n} FlashAttention is not "
                             f"faster than mem_efficient, so the manuscript's "
                             f"'our fused gap is a lower bound' fails")

    # ---- Memory ---------------------------------------------------------
    mem = load("memory_analysis.json")
    tag = next(iter(mem["measured"]))
    ad = mem["measured"][tag]["arms"].get("adaptive", {})
    exp = ad.get("fitted_exponent_offset_corrected")
    checks += 1
    if exp is None or not close(exp, 3.04, 0.02):
        fails.append(f"memory: fitted exponent {exp}, paper says 3.04")

    # ---- Drift ----------------------------------------------------------
    dr = load("drift_report.json")
    idle = next((c for c in dr["comparisons"] if "idle" in c["label"]), None)
    checks += 1
    if idle and not close(idle["ratio_max_abs_pct_diff"], 1.4, 0.05):
        fails.append(f"drift: idle max ratio diff "
                     f"{idle['ratio_max_abs_pct_diff']:.1f}%, paper says 1.4%")

    # ---- Training contrasts --------------------------------------------
    summ = (R / "reduced_design_summary.md")
    if summ.exists():
        s = summ.read_text()
        tr = (PAPER / "05_training.tex").read_text()
        for cid, val in (("C1", "+4.03"), ("C4", "+147.01"),
                         ("C5", "-6.74"), ("C6", "-5.56")):
            checks += 1
            if val.lstrip("+") not in s:
                fails.append(f"training {cid}: {val} not found in summary")
            if val.replace("-", "$-$").replace("+", "$+$") not in tr and \
                    val.lstrip("+") not in tr:
                fails.append(f"training {cid}: {val} not found in manuscript")

    # ---- Prose guards ---------------------------------------------------
    # Four statements were wrong in an earlier version and were corrected
    # after review. They are cheap to reintroduce while editing, so the
    # checker refuses them rather than trusting us to remember.
    prose = {p.name: p.read_text() for p in PAPER.glob("*.tex")}
    banned = [
        ("at most one contrast",
         "Holm is a step-down procedure; it can reject more than one"),
        ("monotonically in $d$",
         "the d=32 -> d=64 step rises, so the head-width trend is not "
         "monotone"),
        ("falls monotonically in head width",
         "the d=32 -> d=64 step rises, so the head-width trend is not "
         "monotone"),
        ("decreases monotonically as $d$ grows",
         "the d=32 -> d=64 step rises, so the head-width trend is not "
         "monotone"),
        ("channel carries information",
         "the positive controls establish an architecture effect, not a "
         "channel-specific one"),
    ]
    for phrase, why in banned:
        checks += 1
        for name, text in prose.items():
            if phrase in text:
                fails.append(f"{name}: reintroduced '{phrase}' -- {why}")

    # The apparent saving the aggregation-only model reports at the headline
    # cell, which the manuscript quotes as 75%.
    checks += 1
    apparent = 1 - 127 / 511
    if not close(apparent, 0.75, 0.01):
        fails.append(f"apparent saving at N=512, k=127 is {apparent:.3f}, "
                     f"paper says 0.75")

    print(f"checked {checks} quantities across "
          f"{len(list(PAPER.glob('*.tex')))} manuscript sections")
    if fails:
        print(f"\n{len(fails)} MISMATCH(ES):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("manuscript agrees with every measurement artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
