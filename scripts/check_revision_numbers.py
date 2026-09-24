"""Verify the revised manuscript against its artifacts, and against its own past.

Three kinds of check, in increasing order of how easy they are to break by
accident:

1. **Tabular numbers.** Every table is generated from JSON by
   ``scripts/make_revision_tables.py``, so the check is that regenerating them
   produces what is on disk. A stale table is then a failure rather than a
   silent disagreement.
2. **Prose numbers.** Numbers that appear in sentences are re-derived from the
   artifacts here, because generation cannot reach them.
3. **Prose regressions.** Statements that were wrong in an earlier version, and
   writing patterns that a previous revision removed, are banned by exact
   phrase. The purpose is regression protection against known text, not a style
   checker: ordinary uses of "first" or "Table" are untouched.

Exit code 1 on any failure.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
PAPER = ROOT / "paper_submission_revision"
SEC = PAPER / "sections"
TAB = PAPER / "tables"


#: Section files set in the main text; the rest are appendices. Sections 05
#: (learning study) and 07 (measurement error) moved to the appendix when the
#: paper was restructured for length.
MAIN_TEXT = ("00_abstract.tex", "01_intro.tex", "02_related.tex",
             "03_analysis.tex", "04_measurement.tex", "08_limitations.tex",
             "09_conclusion.tex")


def load(name: str) -> dict:
    p = R / name
    if not p.exists():
        raise SystemExit(f"missing artifact {p}")
    return json.loads(p.read_text())


def live(pay: dict) -> list[dict]:
    return [r for r in pay["rows"] if not r.get("failed")]


def at(rows, impl, N, k=None):
    c = [r for r in rows if r["impl"] == impl and r["n_agents"] == N
         and (k is None or r["k"] == k)]
    return c[0] if c else None


def ms(r, field="forward"):
    return r[field]["min_ms"] if r and r.get(field) else None


def _word(n: int) -> str:
    """Small integers as the manuscript spells them, so a prose number can be
    checked against an artifact without the prose becoming numerals."""
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen",
             "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen", "twenty"]
    return names[n] if 0 <= n < len(names) else str(n)


def main() -> int:
    fails: list[str] = []
    checks = 0
    prose = {p.name: p.read_text() for p in SEC.glob("*.tex")}
    flat = {n: re.sub(r"\s+", " ", t) for n, t in prose.items()}
    lower = {n: t.lower() for n, t in flat.items()}
    everything = " ".join(flat.values())

    # ---- 1. the generated tables are not stale -------------------------
    before = {p.name: p.read_text() for p in TAB.glob("*.tex")}
    subprocess.run([sys.executable, str(ROOT / "scripts" /
                                        "make_revision_tables.py")],
                   check=True, capture_output=True)
    for name, old in before.items():
        checks += 1
        new = (TAB / name).read_text()
        if old != new:
            fails.append(f"table {name} is stale: regenerating changes it")

    # ---- 2. every implementation computes the same function ------------
    core = load("attention_core_d16.json")
    for impl, dev in core.get("max_deviation_by_impl", {}).items():
        checks += 1
        if not (dev == dev and dev < 1e-4):
            fails.append(f"{impl} deviates from the reference by {dev:.2e}; "
                         "a latency for it is not a latency for this function")

    rows = live(core)

    # ---- 3. prose numbers ----------------------------------------------
    # The crossover claim: there is a k below which the best sparse
    # implementation beats the best dense one, and it is small.
    crossings = {}
    for N in sorted({r["n_agents"] for r in rows}):
        bd = min(x for x in (
            (at(rows, "dense_fused", N) or {}).get("forward", {}).get("min_ms"),
            (at(rows, "dense_triton", N) or {}).get("forward", {}).get("min_ms"),
            (at(rows, "dense_tuned", N) or {}).get("forward", {}).get("min_ms"))
            if x is not None)
        best = {}
        for k in sorted({r["k"] for r in rows if r["impl"] == "topk_gather"
                         and r["n_agents"] == N}):
            cand = [(at(rows, i, N, k) or {}).get("forward", {}).get("min_ms")
                    for i in ("topk_gather", "topk_fused_gather",
                              "topk_fused_masked")]
            cand = [c for c in cand if c is not None]
            if cand:
                best[k] = min(cand) / bd
        crossings[N] = best

    # A sparse implementation faster than dense by more than the operation-count
    # ceiling permits is a signal about the baseline, not a result: the ceiling
    # bounds operations, not time, so this is a smoke alarm and not a proof.
    for N, best in crossings.items():
        for k, ratio in best.items():
            checks += 1
            if ratio < 0.5:
                fails.append(
                    f"N={N}, k={k}: best sparse is {1/ratio:.2f}x the best "
                    "dense. Sparsity cannot supply that at this k; inspect the "
                    "dense baseline's efficiency before reporting it")

    # The manuscript must not claim the sparse path always loses if it does not.
    ever_wins = any(r < 1.0 for b in crossings.values() for r in b.values())
    checks += 1
    # An explicit exception clause -- "at every configuration but one" -- is a
    # scoped claim and not the unconditional one this check is about.
    # Scoped to the sections about the score-then-select path: Section 5's
    # statement that the *composed pre-score* stage is slower everywhere is a
    # different claim about a different implementation, and it is true.
    stselect = " ".join(t for n, t in flat.items()
                        if n.startswith(("01_", "04_", "09_")))
    if ever_wins and re.search(r"slower than (the best )?dense.{0,40}"
                               r"(every|all) (measured )?configuration"
                               r"(?!s? but\b| except\b)",
                               stselect, re.I):
        fails.append("a sparse implementation is faster somewhere, so the "
                     "unconditional 'slower everywhere' claim is false")

    # Prose numbers, each re-derived from the artifact it came from. A number
    # in a sentence cannot be generated the way a table cell can, so it is
    # checked here instead.
    def check_num(desc, value, pattern, tol=0.02):
        nonlocal_checks.append(1)
        m = re.search(pattern, everything)
        if not m:
            fails.append(f"{desc}: no sentence matching {pattern!r}")
            return
        quoted = float(m.group(1))
        if value is None:
            fails.append(f"{desc}: no artifact value to check against")
            return
        if abs(quoted - value) > tol * max(abs(value), 1e-9):
            fails.append(f"{desc}: manuscript says {quoted}, artifact says "
                         f"{value:.4g}")

    nonlocal_checks: list[int] = []
    N_top = max(r["n_agents"] for r in rows)
    comp = (at(rows, "topk_gather", N_top, 1) or {}).get("forward", {}).get("min_ms")
    fus = (at(rows, "topk_fused_gather", N_top, 1) or {}).get("forward", {}).get("min_ms")
    if comp and fus:
        check_num("composed sparse at the largest N, k=1", comp,
                  r"composed sparse path costs \$([\d.]+)\$\\,ms")
        check_num("fused sparse at the largest N, k=1", fus,
                  r"the fused one \$([\d.]+)\$\\,ms")
        check_num("composed-to-fused implementation factor", comp / fus,
                  r"a factor of \$(\d+)\$ attributable to the implementation",
                  tol=0.05)

    st = load("stage_accounting.json")["rows"]
    sN = max(r["n_agents"] for r in st)

    def frac(stage):
        c = [r for r in st if r["stage"] == stage and r["n_agents"] == sN]
        return 100 * c[0]["frac_of_binding_roof"] if c else None

    check_num("sparse aggregation attainment", frac("aggregate_sparse"),
              r"\$(\d+)\\%\$ for the sparse contraction", tol=0.03)
    check_num("dense aggregation attainment", frac("aggregate_dense"),
              r"\$(\d+)\\%\$ for the dense multiply", tol=0.03)

    # The two dense baselines' gap is quoted in Section 7 and in its table. It
    # was once quoted from the superseded microbenchmark, where the same pair
    # differs by 4.6x rather than the 5.4x this campaign measures.
    du = (at(rows, "dense_unfused", N_top) or {}).get("forward", {}).get("min_ms")
    df = (at(rows, "dense_fused", N_top) or {}).get("forward", {}).get("min_ms")
    if du and df:
        check_num("unfused-to-fused dense gap at the largest N", du / df,
                  r"unfused and fused dense paths differ by \$([\d.]+)\\times\$",
                  tol=0.03)
    # ---- the two kernels written to answer the review ---------------------
    # Both exist to test a claim the manuscript previously made without one, so
    # every number they put into prose is re-derived here from their artifacts.
    srt = load("attention_core_sort.json")
    srows = live(srt)

    def sms(impl, N, k=None):
        return ms(at(srows, impl, N, k))

    sort_lo = sms("topk_fused_sort", 192, 1)
    sort_hi = sms("topk_fused_sort", 192, 16)
    arg_lo = sms("topk_fused_gather", 192, 1)
    arg_hi = sms("topk_fused_gather", 192, 16)
    if sort_lo and sort_hi:
        check_num("sorted kernel at one retained peer, 192 agents", sort_lo,
                  r"it runs in \$([\d.]+)\$\\,ms at one retained peer")
        check_num("sorted kernel at sixteen retained peers, 192 agents",
                  sort_hi, r"and \$([\d.]+)\$\\,ms at sixteen")
        # The claim is that what grows with k is the aggregation and not the
        # selection, so what is checked is the size of the two growths, not
        # flatness -- the kernel is not flat and the manuscript no longer says
        # it is.
        checks += 1
        rise = round(100 * (sort_hi / sort_lo - 1))
        if f"a rise of {_word(rise)} per cent" not in everything.lower():
            fails.append("the manuscript does not state the sorted kernel's "
                         f"rise from one to sixteen retained peers as "
                         f"{_word(rise)} per cent")
        checks += 1
        if sort_hi / sort_lo > 1.25:
            fails.append("the sorted kernel's cost rises by "
                         f"{100 * (sort_hi / sort_lo - 1):.0f}% from one to "
                         "sixteen retained peers, which is not a small rise "
                         "and the sentence describing it says it is")
        checks += 1
        if arg_lo and arg_hi and arg_hi / arg_lo < 5:
            fails.append("the argmax kernel is described as rising eightfold "
                         f"over the same range; measured {arg_hi / arg_lo:.1f}")
        bd192 = min(x for x in (sms("dense_fused", 192), sms("dense_triton", 192),
                                sms("dense_tuned", 192)) if x is not None)
        checks += 1
        if not 4.5 <= sort_hi / bd192 <= 5.6:
            fails.append("the sorted kernel is described as five times dense at "
                         f"192 agents; measured {sort_hi / bd192:.2f}")
        cross = [k for k in (1, 2, 4, 8, 16, 32)
                 if sms("topk_fused_sort", 192, k) is not None
                 and sms("topk_fused_sort", 192, k) < sms("topk_fused_gather", 192, k)]
        checks += 1
        if not cross or min(cross) != 16:
            fails.append("the manuscript says the sorted kernel overtakes the "
                         "argmax kernel at sixteen retained peers; the "
                         f"artifact puts it at {min(cross) if cross else 'none'}")
        big = [sms("topk_fused_sort", N, k) / min(
                   x for x in (sms("dense_fused", N), sms("dense_triton", N),
                               sms("dense_tuned", N)) if x is not None)
               for N in (384, 512) for k in (1, 2, 4, 8, 16)]
        checks += 1
        if not (12.5 <= min(big) and max(big) <= 16.5):
            fails.append("the manuscript says thirteen to sixteen times dense "
                         f"at 384 and 512 agents; measured {min(big):.1f} to "
                         f"{max(big):.1f}")

    pre = load("prescore.json")["rows"]

    def pat(sel, N, k):
        c = [r for r in pre if r["selector"] == sel and r["n_agents"] == N
             and r["k"] == k]
        return c[0] if c else None

    pns = sorted({r["n_agents"] for r in pre})
    pks = sorted({r["k"] for r in pre})
    for k, phrase in ((pks[0], r"[Aa]t two retained peers the attention stage "
                               r"costs \$([\d.]+)\$"),
                      (pks[1], r"at four, \$([\d.]+)\$")):
        rowset = [pat("morton_window", N, k) for N in pns]
        if all(rowset):
            att = [r["t_attention_fused_ms"] / r["t_dense_best_ms"]
                   for r in rowset]
            check_num(f"fused pre-score attention at {pns[0]} agents, k={k}",
                      att[0], phrase, tol=0.03)
    mz = [pat("morton_window", N, pks[1]) for N in pns]
    if all(mz):
        tot = [r["t_total_fused_ms"] / r["t_dense_best_ms"] for r in mz]
        check_num("fused pre-score total at the largest N", tot[-1],
                  r"\$([\d.]+)\$ times dense attention from \$48\$ to \$512\$ agents\.",
                  tol=0.03)
        checks += 1
        if tot[-1] >= 1.0:
            fails.append("the manuscript says the fused pre-score path is "
                         f"faster end to end at {pns[-1]} agents; it is "
                         f"{tot[-1]:.2f} times dense")
    # The crossing is claimed for the two sparsest settings only, so where it
    # stops is checked rather than described.
    # The end-to-end crossing is claimed on minima and explicitly withdrawn
    # under the paired median, so both estimators are checked here: the sign
    # under minima, and the fact that it reverses under medians.
    def ratio(r, series, field):
        den = min(v[field] for v in r["dense_dispersion"].values())
        return r["dispersion"][series][field] / den

    r512 = pat("morton_window", pns[-1], pks[0])
    checks += 1
    if ratio(r512, "total_fused", "min_ms") >= 1.0:
        fails.append("the manuscript claims the pre-score path reaches parity "
                     "at the largest agent count on minima and it does not")
    check_num("pre-score total under minima at the largest N",
              ratio(r512, "total_fused", "min_ms"),
              r"path costs \$([\d.]+)\$ times dense at two", tol=0.03)
    check_num("pre-score total under the paired median at the largest N",
              ratio(r512, "total_fused", "median_ms"),
              r"it costs \$([\d.]+)\$\s*times dense instead", tol=0.03)
    checks += 1
    if ratio(r512, "total_fused", "median_ms") < 1.0:
        fails.append("the manuscript says the end-to-end crossing reverses "
                     "under the paired median and it does not")
    r8 = pat("morton_window", pns[-1], pks[2])
    check_num("pre-score total under minima at eight retained peers",
              ratio(r8, "total_fused", "min_ms"),
              r"and \$([\d.]+)\$ at eight", tol=0.03)
    # The attention stage is claimed as robust to the estimator, so the claim
    # that it is robust is itself checked.
    b_min = min(ratio(r, "fused", "min_ms") for r in pre)
    b_med = min(ratio(r, "fused", "median_ms") for r in pre)
    check_num("attention stage under minima", 1 / b_min,
              r"is \$([\d.]+)\$ times faster than dense on minima", tol=0.02)
    check_num("attention stage under the paired median", 1 / b_med,
              r"on minima and \$([\d.]+)\$ on\s*medians", tol=0.02)
    best = min(r["t_attention_fused_ms"] / r["t_dense_best_ms"] for r in pre)
    check_num("best fused pre-score attention over the grid", best,
              r"best case over the grid is \$([\d.]+)\$", tol=0.03)
    for sel, phrase in (
            ("morton_window", r"the stage costs \$([\d.]+)\$ times\s*dense"),
            ("radius_knn", r"Z-order selector and \$([\d.]+)\$ behind")):
        worst = max(r["t_attention_fused_ms"] / r["t_dense_best_ms"]
                    for r in pre if r["selector"] == sel)
        check_num(f"worst fused pre-score attention, {sel}", worst, phrase,
                  tol=0.03)
    comp = [r["t_attention_ms"] / r["t_dense_best_ms"] for r in pre]
    checks += 1
    if min(comp) < 1.0:
        fails.append("the manuscript says the composed pre-score stage is "
                     "slower than dense at every configuration and it is not")
    check_num("composed pre-score attention, best", min(comp),
              r"configurations, by \$([\d.]+)\$ to", tol=0.03)
    check_num("composed pre-score attention, worst", max(comp),
              r"configurations, by \$[\d.]+\$ to \$([\d.]+)\$", tol=0.03)
    fac = [r["t_attention_ms"] / r["t_attention_fused_ms"] for r in pre]
    check_num("fusing factor, smallest", min(fac),
              r"is worth between \$([\d.]+)\$", tol=0.03)
    check_num("fusing factor, largest", max(fac),
              r"is worth between \$[\d.]+\$ and \$([\d.]+)\$", tol=0.03)
    sel_ms = [r["t_selector_ms"] for r in pre if r["selector"] == "morton_window"]
    check_num("Z-order selector, cheapest", min(sel_ms),
              r"nearly flat in the agent count --- \$([\d.]+)\$ to", tol=0.03)
    check_num("Z-order selector, dearest", max(sel_ms),
              r"nearly flat in the agent count --- \$[\d.]+\$ to \$([\d.]+)\$\\,ms",
              tol=0.03)
    knn = [pat("radius_knn", N, pks[1]) for N in pns]
    if all(knn):
        kt = [r["t_total_fused_ms"] / r["t_dense_best_ms"] for r in knn]
        check_num("nearest-neighbour path at the largest N", max(kt),
                  r"rises to \$([\d.]+)\$ times dense", tol=0.03)
        checks += 1
        if kt[-1] < kt[0]:
            fails.append("the nearest-neighbour path is described as getting "
                         "worse with scale and it does not")

    # The pre-score sweep is run twice, and the crossing it reports is a margin
    # of about ten per cent, so the manuscript states how far the two runs
    # agree at the cells that carry it.
    rep = R / "prescore_repeat.json"
    checks += 1
    if not rep.exists():
        fails.append("results/prescore_repeat.json is missing; the pre-score "
                     "crossing is quoted with a reproducibility bound")
    else:
        b = {(r["selector"], r["n_agents"], r["k"]): r
             for r in json.loads(rep.read_text())["rows"]}
        worst_rep = 0.0
        for k in pks:
            r1 = pat("morton_window", pns[-1], k)
            r2 = b[("morton_window", pns[-1], k)]
            x1 = r1["t_total_fused_ms"] / r1["t_dense_best_ms"]
            x2 = r2["t_total_fused_ms"] / r2["t_dense_best_ms"]
            worst_rep = max(worst_rep, abs(x1 - x2) / min(x1, x2))
        checks += 1
        if 100 * worst_rep > 2.5:
            fails.append("the manuscript says the two runs agree to within "
                         f"2.3 per cent at the crossing cells; measured "
                         f"{100 * worst_rep:.1f}%")

    # ---- the second pre-score sweep: three selectors and FlexAttention ----
    ps = load("prescore_selectors.json")["rows"]

    def psel(sel, N, k):
        c = [r for r in ps if r["selector"] == sel and r["n_agents"] == N
             and r["k"] == k]
        return c[0] if c else None

    def pden(r, f="min_ms"):
        return min(v[f] for v in r["dense_dispersion"].values())

    a192, a512 = psel("radius_knn", 192, 4), psel("radius_knn", 512, 4)
    g192, g512 = psel("grid_knn", 192, 4), psel("grid_knn", 512, 4)
    z512 = psel("morton_window", 512, 4)
    check_num("all-pairs selector at 192 agents", a192["t_selector_ms"],
              r"quadratically, from \$([\d.]+)\$ to", tol=0.03)
    check_num("all-pairs selector at 512 agents", a512["t_selector_ms"],
              r"quadratically, from \$[\d.]+\$ to \$([\d.]+)\$", tol=0.03)
    check_num("all-pairs path at 512 agents",
              a512["t_total_fused_ms"] / pden(a512),
              r"its path ends \$([\d.]+)\$ times slower", tol=0.03)
    check_num("grid selector at 192 agents", g192["t_selector_ms"],
              r"cost rises from \$([\d.]+)\$ to", tol=0.03)
    check_num("grid selector at 512 agents", g512["t_selector_ms"],
              r"cost rises from \$[\d.]+\$ to \$([\d.]+)\$", tol=0.03)
    check_num("grid path at 512 agents", g512["t_total_fused_ms"] / pden(g512),
              r"keeps the path \$([\d.]+)\$ times\s*slower", tol=0.05)
    check_num("Z-order path at 512 agents, minima",
              z512["t_total_fused_ms"] / pden(z512),
              r"the path costs \$([\d.]+)\$ times dense at \$512\$", tol=0.03)
    check_num("Z-order path at 512 agents, medians",
              z512["dispersion"]["total_fused"]["median_ms"]
              / pden(z512, "median_ms"),
              r"in the second sweep \(\$([\d.]+)\$\)", tol=0.03)
    check_num("Z-order through FlexAttention at 512 agents",
              z512["t_flex_total_ms"] / pden(z512),
              r"window costs \$([\d.]+)\$ times dense", tol=0.04)
    check_num("grid recall, worst", 100 * min(
        r["recall_vs_exact_knn"] for r in ps if r["selector"] == "grid_knn"),
              r"at least\s*\$([\d.]+)\\%\$", tol=0.0001)
    for sel in ("radius_knn", "grid_knn"):
        checks += 1
        r16 = [psel(sel, N, 16) for N in (384, 512)]
    checks += 1
    if not all(psel("grid_knn", N, 16)["t_total_fused_ms"]
               > psel("radius_knn", N, 16)["t_total_fused_ms"]
               for N in (384, 512)):
        fails.append("the grid selector is said to be slower than all-pairs "
                     "at sixteen retained peers at 384 and 512 agents")
    checks += 1
    zf = [r for r in ps if r["selector"] == "morton_window"]
    if not all(r["flex_deviation"] < 1e-4 for r in zf):
        fails.append("FlexAttention over the Z-order band does not compute "
                     "the same function as the pre-score path")

    # The equivalence appendix quotes the artifact, so the artifact checks it.
    eq = load("equivalence.json")
    checks += 1
    if eq.get("failures"):
        fails.append(f"{eq['failures']} implementations disagree with the "
                     "reference; no latency in the paper is a latency for the "
                     "reference function")
    check_num("configurations checked", eq["configurations"],
              r"\$(\d+)\$ combinations of agent count")
    check_num("configurations exercising selection",
              eq["configurations_exercising_selection"],
              r"\$(\d+)\$ of them with \$k < N-1\$")
    check_num("query rows checked", eq["query_rows_checked"] / 1e6,
              r"over \$([\d.]+)\$ million query rows", tol=0.01)
    # Spelled out in the prose, as the house style requires for small counts,
    # so the check is on the word rather than on a captured number.
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    ties_n = eq["query_rows_with_exact_tie_at_k"]
    checks += 1
    if f"{words.get(ties_n, ties_n)} query rows in" not in everything.lower():
        fails.append(f"the appendix does not say that {ties_n} query rows have "
                     "an exact tie at the k-th score")
    tie_cfg = eq["configurations_with_tie_divergence"]
    checks += 1
    if f"at {words.get(tie_cfg, tie_cfg)} of the" not in everything.lower():
        fails.append(f"the appendix does not say that the tie divergences fall "
                     f"at {tie_cfg} configurations")
    check_num("pre-score configurations checked",
              eq["prescore"]["configurations"],
              r"at all \$(\d+)\$ configurations")
    checks += 1
    if eq["prescore"]["failures"]:
        fails.append("the fused pre-score kernel disagrees with the composed "
                     "path it is compared against")

    checks += len(nonlocal_checks)

    # ---- 3b. no macro damaged into literal text -------------------------
    # An escaped backslash in an editing script turns \ref into a carriage
    # return and the word "ef", which LaTeX sets as text and reports as no
    # error at all. Four such wrecks reached a compiled PDF once.
    for src in sorted(SEC.glob("*.tex")) + sorted(TAB.glob("*.tex")):
        raw = src.read_bytes()
        checks += 1
        if b"\r" in raw or b"\t" in raw:
            fails.append(f"{src.name}: contains a carriage return or tab, "
                         "which is how an escaped LaTeX macro arrives")
        text = raw.decode("utf8")
        for frag in ("ef{", "extbf{", "exttt{", "extit{", "egin{", "rac{"):
            checks += 1
            for m in re.finditer(re.escape(frag), text):
                # A surviving macro keeps its name letter in front of the
                # fragment (\ref, \begin); a wrecked one has whitespace or a
                # control character there, because the escape ate the letter.
                prev = text[m.start() - 1] if m.start() else " "
                if not prev.isalpha():
                    fails.append(f"{src.name}: '{frag}' at offset {m.start()} "
                                 "is a LaTeX macro that lost its backslash")

    # ---- 4. banned prose ------------------------------------------------
    banned = [
        # statements corrected after review
        ("at most one contrast",
         "Holm is a step-down procedure; it can reject more than one"),
        ("neither trained arm beats a uniform-random policy",
         "the gathered variant at twelve agents beats the random baseline"),
        ("neither variant beats a uniform-random policy",
         "the gathered variant at twelve agents beats the random baseline"),
        ("$5\\times10^{-4}$",
         "the margin between 2/252 and 0.05/6 is 4e-4, not 5e-4"),
        ("channel carries information",
         "the positive controls establish an architecture effect, not a "
         "channel-specific one"),
        ("batched matrix--vector product has far lower arithmetic intensity",
         "measured, the sparse contraction attains a higher fraction of "
         "bandwidth than the dense matmul; it is slower because it moves "
         "more bytes"),
        ("the same tiling space",
         "the two searches admit very unequal configuration counts; say "
         "'one parameter space under one admissibility filter'"),
        ("with the same budget",
         "the sparse search admits three configurations above 192 agents "
         "against the dense search's 28 to 106"),
        ("$4.6\\times$",
         "the 4.6x dense-baseline gap is from the superseded "
         "microbenchmark; this campaign measures 5.4x"),
        ("could not write a kernel with both",
         "the sorted kernel has sparse aggregation and k-independent "
         "selection at once; it is simply not faster"),
        ("sparse arithmetic and $k$-independent selection",
         "the sorted kernel obtains both, so this is no longer an open gap"),
        ("at best fifteen per cent",
         "fused, the pre-score attention stage is up to 18x faster than the "
         "best dense implementation"),
        ("loses the matrix-multiply shape",
         "the composed gathered form was the cost, not the shape; a fused "
         "kernel over the same gathered peers is 18x faster than dense"),
        ("not a shape this hardware is quick at",
         "refuted by the fused pre-score kernel"),
        ("not a shape the hardware multiplies quickly",
         "refuted by the fused pre-score kernel"),
        ("pre-score attention stage is never faster",
         "fused, it is faster than the best dense implementation at every "
         "agent count measured"),
        ("neither delivers a speedup",
         "the Z-order path is faster end to end at 512 agents"),
        ("slower than unfused dense in all $46$",
         "the campaign no longer has 46 cells and the claim is now "
         "conditional on k"),

        # writing patterns a previous revision removed
        ("first, an analytic ceiling", "outline prose in the abstract"),
        ("second, what executes", "outline prose in the abstract"),
        ("third, what it costs", "outline prose in the abstract"),
        ("constructively:", "outline prose in the abstract"),
        ("table 2 reports the outcome", "artifact narration"),
        ("table 1 says that", "artifact narration"),
        ("gives the numbers behind", "artifact narration"),
        ("gives the per-stage numbers behind", "artifact narration"),
        ("p1, p3, p6 and p7", "opaque identifiers as prose subjects"),
        ("is the contrast the design exists for",
         "name what is compared, not its internal identifier"),
        ("we built one and report why we abandoned it", "lab-diary framing"),
        ("a report on why we abandoned that design", "lab-diary framing"),
        ("a reader is entitled to", "meta-commentary about candour"),
        ("we should be direct", "meta-commentary about candour"),
        ("we state this plainly", "meta-commentary about candour"),
        ("rather than taken on trust", "meta-commentary about candour"),
        ("we withdraw any suggestion", "meta-commentary about candour"),
        ("cannot rank what it has not scored", "slogan"),
        ("within the analytic ceiling",
         "a MAC-count bound does not bound wall-clock in either direction"),
        ("inside the analytic ceiling",
         "a MAC-count bound does not bound wall-clock in either direction"),
        ("with implementation held constant",
         "the fused sparse and dense kernels are different kernels"),
        ("the archival version",
         "the submitted paper should not narrate its own revision history"),
        ("the manuscript's earlier version",
         "the submitted paper should not narrate its own revision history"),
        ("version of this manuscript",
         "the submitted paper should not narrate its own revision history"),
        ("earlier version of this work",
         "the submitted paper should not narrate its own revision history"),
        ("we report the correction rather than the claim",
         "report the corrected number; do not narrate having reported it"),
        ("we report it as the null it is",
         "report the null; do not narrate having reported it"),
        ("decide where to look before looking", "slogan"),
        ("the arm meant to save cost", "slogan, and 'arm' is harness jargon"),
        ("one and a half to twenty-eight",
         "from four retained peers upward the range is 1.3 to 28"),
        ("narrow head widths, and nowhere else",
         "at d=32, N=512 sparsity also pays at two retained peers"),
        ("by less than a fifth",
         "the batch sweep moves the ratio by up to a third at N=512"),
        ("every measurement in this paper was at some point wrong",
         "self-damaging framing a reviewer will quote"),
        ("one we proposed and built",
         "de-anonymises the submission"),
        ("commit history establishes",
         "points reviewers at a de-anonymising repository"),
        ("a ratio of minima has no sampling distribution",
         "false: a bootstrap distribution exists"),
    ]
    for phrase, why in banned:
        checks += 1
        for name, text in lower.items():
            if phrase.lower() in text:
                fails.append(f"{name}: banned phrase {phrase!r} -- {why}")

    # Case-sensitive: Corollary 1(d) needs little-o, since C_sel = O(Nkd)
    # gives Theta(N/k) and not S ~ N/k. Lower-casing would hide the difference.
    checks += 1
    if "C_{\\mathrm{sel}} = O(Nkd)$ and in particular" in flat.get(
            "03_analysis.tex", ""):
        fails.append("03_analysis.tex: Corollary 1(d) states O(Nkd) where "
                     "S ~ N/k needs o(Nkd)")

    # The abstract carries no mathematical notation.
    checks += 1
    abstract = prose.get("00_abstract.tex", "")
    if "$" in abstract or "\\(" in abstract:
        fails.append("00_abstract.tex: the abstract must contain no "
                     "mathematical notation")

    # Harness vocabulary must not carry the narrative.
    for word, why in (("arm", "say implementation"),
                      ("cell", "say configuration"),
                      ("timing group", "say interleaved sweep")):
        checks += 1
        hits = sum(len(re.findall(rf"\b{word}s?\b", t, re.I))
                   for n, t in flat.items() if n in MAIN_TEXT)
        if hits > 2:
            fails.append(f"harness vocabulary {word!r} appears {hits} times "
                         f"in the main text -- {why}")

    # Corollary 1's worked example: the Z-order closed form at N=512, k=4, d=16
    # and its fraction of N/k, which an earlier revision misquoted as 72%.
    N, k, d = 512, 4, 16
    # The selector charges floor(log2 N) + 1 = N.bit_length() sort levels,
    # as src/kernels/prescore_selection.analytic_flops does.
    zorder = 4 * N * d / (4 * k * d + 30 + 2 * N.bit_length())
    knn = N * d / (k * d + N)
    for want, why in ((f"{knn:.1f}", "kNN closed form"),
                      (f"${zorder:.0f}$", "Z-order closed form"),
                      (f"${100 * zorder / (N / k):.0f}\\%$",
                       "Z-order fraction of N/k")):
        checks += 1
        if want not in everything:
            fails.append(f"manuscript: expected {want!r} ({why})")

    # ---- 5. required scoping --------------------------------------------
    required = [
        ("forward pass",
         "the twofold ceiling holds for the forward pass only"),
        ("score-then-select",
         "Proposition 1 must be scoped to score-then-select rules"),
        ("centralised inference",
         "the cost claims must be scoped away from distributed deployment"),
        ("shared with other tenants",
         "the measurement envelope must disclose the shared device"),
    ]
    for phrase, why in required:
        checks += 1
        if phrase.lower() not in everything.lower():
            fails.append(f"missing required scoping {phrase!r} -- {why}")

    # ---- report ---------------------------------------------------------
    print(f"checked {checks} conditions across "
          f"{len(prose)} manuscript sections and {len(before)} tables")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("manuscript agrees with every measurement artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
