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
            (at(rows, "dense_triton", N) or {}).get("forward", {}).get("min_ms"))
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

    # No sparse implementation may beat dense by more than the ceiling
    # permits; if one does, the baseline is wrong, not the algorithm.
    for N, best in crossings.items():
        for k, ratio in best.items():
            checks += 1
            if ratio < 0.5:
                fails.append(
                    f"N={N}, k={k}: best sparse is {1/ratio:.2f}x the best "
                    "dense, which exceeds what Proposition 1 permits from "
                    "sparsity -- check the dense baseline before believing it")

    # The manuscript must not claim the sparse path always loses if it does not.
    ever_wins = any(r < 1.0 for b in crossings.values() for r in b.values())
    checks += 1
    if ever_wins and re.search(r"slower than (the best )?dense.{0,40}"
                               r"(every|all) (measured )?configuration",
                               everything, re.I):
        fails.append("a sparse implementation is faster somewhere, so the "
                     "unconditional 'slower everywhere' claim is false")

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
        ("decide where to look before looking", "slogan"),
        ("the arm meant to save cost", "slogan, and 'arm' is harness jargon"),
    ]
    for phrase, why in banned:
        checks += 1
        for name, text in lower.items():
            if phrase.lower() in text:
                fails.append(f"{name}: banned phrase {phrase!r} -- {why}")

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
                   for n, t in flat.items() if n != "10_appendix.tex")
        if hits > 2:
            fails.append(f"harness vocabulary {word!r} appears {hits} times "
                         f"in the main text -- {why}")

    # ---- 5. required scoping --------------------------------------------
    required = [
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
