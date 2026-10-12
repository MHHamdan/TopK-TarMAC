"""A manifest of every number quoted in the main text, and its verification.

``check_revision_numbers.py`` searches the whole manuscript for a pattern and
checks the first match, so a number repeated in the abstract and an appendix is
verified only where it appears first, and spelled-out numbers ("thirty-seven")
are not seen at all. This manifest closes both gaps for the main text. Each
entry names the section file the number appears in, a pattern that captures it
there, the artifact it is derived from, the configuration it describes, and a
tolerance; verification fails if the pattern is missing from that file or the
captured value disagrees with the artifact.

Run ``python scripts/check_revision_numbers.py --verify-paper-numbers``, or this
module directly, which also writes ``results/paper_numbers.json``.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SEC = ROOT / "paper_submission_revision" / "sections"

from scripts.make_revision_tables import (  # noqa: E402
    at, best_dense, crossover_replay, live, load, ms,
    streaming_find_share, streaming_macros)

WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    .split())}
WORDS.update({"thirty-seven": 37, "nine tenths": 0.9, "a quarter": 0.25})


def _num(tok: str) -> float:
    tok = tok.strip().replace("{,}", "").replace(",", "")
    return float(WORDS[tok]) if tok in WORDS else float(tok)


@dataclass
class Number:
    id: str
    file: str
    pattern: str            # regex; each group captures one number
    value: Callable[[], float | tuple[float, ...]]
    unit: str
    configuration: str
    source: str
    tol: float = 0.02       # relative, after rounding to the printed precision


# --------------------------------------------------------------------------
# artifact accessors
# --------------------------------------------------------------------------

def _core():
    return live(load("attention_core_d16.json"))


def core_ratio(impl: str, N: int, k: int) -> float:
    rows = _core()
    return ms(at(rows, impl, N, k)) / best_dense(rows, N)


def best_fused_k4_range() -> tuple[float, float]:
    """Fused score-then-select at four retained peers, fp32: the precision
    sweep at d in {16, 32} and the main sweep at d=16, N >= 192."""
    vals = [r["timed"]["fused_topk:4"]["min_ms"] / r["best_dense_ms"]
            for r in _precision("fp32")]
    vals += [core_ratio("topk_fused_gather", N, 4) for N in (192, 384, 512)]
    return min(vals), max(vals)


def _precision(mode: str | None = None):
    return [r for r in load("precision.json")["rows"]
            if mode is None or r["mode"] == mode]


def _prec_k1(mode: str, est: str = "min_ms") -> list[float]:
    out = []
    for r in _precision(mode):
        d = r["best_dense_ms"] if est == "min_ms" else r["best_dense_median_ms"]
        out.append(r["timed"]["fused_topk:1"][est] / d)
    return out


def prec_k1_range(mode_set, Ns) -> tuple[float, float]:
    v = [r["timed"]["fused_topk:1"]["min_ms"] / r["best_dense_ms"]
         for r in load("precision.json")["rows"]
         if r["mode"] in mode_set and r["n_agents"] in Ns]
    return min(v), max(v)


def dense_gain(mode: str) -> tuple[float, float]:
    rows = load("precision.json")["rows"]
    g = []
    for r in rows:
        if r["mode"] != mode:
            continue
        f = [x for x in rows if x["mode"] == "fp32"
             and x["head_dim"] == r["head_dim"]
             and x["n_agents"] == r["n_agents"]][0]
        g.append(f["best_dense_ms"] / r["best_dense_ms"])
    return min(g), max(g)


def flex_topk_range() -> tuple[float, float]:
    v = [r["timed"]["flex_topk:1"]["min_ms"] / r["best_dense_ms"]
         for r in _precision()]
    return min(v), max(v)


def flex_dense_wins() -> float:
    rows = [r for r in _precision() if r["mode"] != "fp32"]
    return float(sum(r["best_dense"] == "flex_dense" for r in rows))


def bf16_median_parity() -> float:
    r = [x for x in _precision("bf16") if x["head_dim"] == 16
         and x["n_agents"] == 512][0]
    return r["timed"]["fused_topk:1"]["median_ms"] / r["best_dense_median_ms"]


def _fixed(mode="graph"):
    return [g for g in load("fixed_set.json")["groups"]
            if g["mode"] == mode and g["n_agents"] >= 192]


def fixed(name, d, N, k, mode="graph") -> float:
    g = [x for x in _fixed(mode) if x["head_dim"] == d and x["n_agents"] == N
         and x["k"] == k][0]
    return g["ratios"][name]["ratio_of_minima"]


def fixed_share_min() -> float:
    ks = (1, 2, 4, 16, 32)
    return min(1 - fixed("fixed_set", g["head_dim"], g["n_agents"], g["k"])
               / fixed("fused_stsel", g["head_dim"], g["n_agents"], g["k"])
               for g in _fixed() if g["n_agents"] >= 384 and g["k"] in ks)


def find_set_range() -> tuple[float, float]:
    v = [g["ratios"]["find_set"]["ratio_of_minima"] for g in _fixed()
         if g["head_dim"] == 16]
    return min(v), max(v)


def eager_graph_range() -> tuple[float, float]:
    v = []
    for g in _fixed("eager"):
        if g["k"] > 4:
            continue
        v.append(g["ratios"]["fixed_set"]["ratio_of_minima"]
                 / fixed("fixed_set", g["head_dim"], g["n_agents"], g["k"]))
    return min(v), max(v)


def _pre(name, sel, N, k):
    return [r for r in load(name)["rows"] if r["selector"] == sel
            and r["n_agents"] == N and r["k"] == k][0]


def pre_total(name, sel, N, k, est="min"):
    r = _pre(name, sel, N, k)
    if est == "min":
        return r["t_total_fused_ms"] / r["t_dense_best_ms"]
    d = min(v["median_ms"] for v in r["dense_dispersion"].values())
    return r["dispersion"]["total_fused"]["median_ms"] / d


def pre_attn(name, N, k, est="min"):
    r = _pre(name, "morton_window", N, k)
    if est == "min":
        return r["t_attention_fused_ms"] / r["t_dense_best_ms"]
    d = min(v["median_ms"] for v in r["dense_dispersion"].values())
    return r["dispersion"]["fused"]["median_ms"] / d


def fuse_gain_range() -> tuple[float, float]:
    v = [r["t_attention_ms"] / r["t_attention_fused_ms"]
         for r in load("prescore.json")["rows"]]
    return min(v), max(v)


def sel_ms(sel, N, k=4) -> float:
    return _pre("prescore_selectors.json", sel, N, k)["t_selector_ms"]


def zorder_sel_range() -> tuple[float, float]:
    v = [r["t_selector_ms"] for r in load("prescore_selectors.json")["rows"]
         if r["selector"] == "morton_window"]
    return min(v), max(v)


def grid_recall_min() -> float:
    return 100 * min(r["recall_vs_exact_knn"]
                     for r in load("prescore_selectors.json")["rows"]
                     if r["selector"] == "grid_knn")


def flex_band() -> float:
    r = _pre("prescore_selectors.json", "morton_window", 512, 4)
    return r["t_flex_total_ms"] / r["t_dense_best_ms"]


def env_share_range() -> tuple[float, float]:
    rows = load("training_step_timing.json")["rows"]
    v = [r["phase_share"]["environment"] + r["phase_share"]["policy_forward"]
         for r in rows if r["n_agents"] in (6, 24)]
    return 100 * min(v), 100 * max(v)


def impl_spread_max() -> float:
    rows = load("training_step_timing.json")["rows"]
    out = 0.0
    for N in {r["n_agents"] for r in rows}:
        it = [r["iteration_ms"] for r in rows if r["n_agents"] == N]
        out = max(out, 100 * (max(it) - min(it)) / min(it))
    return out


def autotune_eff(kind: str) -> tuple[float, float]:
    pay = load("autotune.json")
    peak = pay["peak_flops"]
    v = []
    for d in (16, 32):
        for N in (192, 384, 512):
            dn = [r for r in pay["rows"] if r["kind"] == "dense"
                  and r["head_dim"] == d and r["n_agents"] == N][0]
            B, d_ms = dn["batch"], dn["best"]["ms"]
            if kind == "dense":
                v.append(100 * 4 * B * N * N * d / (d_ms * 1e-3) / peak)
                continue
            sp = [r for r in pay["rows"] if r["kind"] == "sparse"
                  and r["head_dim"] == d and r["n_agents"] == N]
            b = min(sp, key=lambda r: r["best"]["ms"])
            k, s_ms = b["k"], b["best"]["ms"]
            v.append(100 * (2 * B * N * N * d + 2 * B * N * k * d)
                     / (s_ms * 1e-3) / peak)
    return min(v), max(v)


def autotune_cfgs(kind: str, Ns=(384, 512)) -> tuple[float, float]:
    v = [r["n_ran"] for r in load("autotune.json")["rows"]
         if r["kind"] == kind and r["n_agents"] in Ns]
    return float(min(v)), float(max(v))


def backward_ratio(field: str) -> float:
    rows = _core()
    g = at(rows, "topk_gather", 512, 128)
    d = at(rows, "dense_fused", 512, 128) or at(rows, "dense_fused", 512)
    return g[field]["min_ms"] / d[field]["min_ms"]


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

R = r"([\d.]+)"


def macro(name: str) -> str:
    return streaming_macros()[name]

def port_pre_attn(N: int = 512, k: int = 2) -> tuple[float, float]:
    """Dense over fused pre-score attention (a speedup factor) on the GPUs of
    results/portability/, at the setting the abstract quotes for Blackwell."""
    v = []
    for p in sorted((ROOT / "results" / "portability").glob("*/prescore.json")):
        r = [x for x in json.loads(p.read_text())["rows"]
             if x["selector"] == "morton_window" and x["n_agents"] == N
             and x["k"] == k][0]
        v.append(r["t_dense_best_ms"] / r["t_attention_fused_ms"])
    return min(v), max(v)


def bf16_k1_blackwell() -> tuple[float, float]:
    from scripts.make_revision_tables import streaming_precision_macros
    m = streaming_precision_macros()
    return float(m["SPBfOneLo"]), float(m["SPBfOneHi"])


MANIFEST: list[Number] = [
    # abstract
    Number("abs.thirty_seven", "00_abstract.tex", r"up to (\d+) times, because",
           lambda: core_ratio("topk_gather", 512, 128), "x",
           "composed gathered, N=512, k=128, d=16", "attention_core_d16.json",
           0.03),
    Number("abs.k4_parity", "00_abstract.tex", r"within\s+(\d+)\\% of parity",
           lambda: float(macro("StreamKFourDev")), "%",
           "streaming k=4, fp32, N<=512, graph replay", "streaming_topk.json", 0),
    Number("abs.find_share", "00_abstract.tex",
           r"at least (\d+)\\% of its time",
           lambda: 100 * streaming_find_share()[0], "%",
           "streaming kernel, graph replay, N in {384,512}, d in {16,32}, min over k",
           "streaming_topk.json", 0.02),
    Number("abs.sixteen", "00_abstract.tex", r"up to (\d+) times\s+faster than dense",
           lambda: 1 / pre_attn("prescore.json", 512, 2), "x",
           "pre-score fused attention, N=512, k=2", "prescore.json", 0.06),
    Number("abs.bf16_k1", "00_abstract.tex",
           r"Blackwell card \(([\d.]+) to ([\d.]+) times dense\)",
           bf16_k1_blackwell, "x", "streaming bf16 k=1, graph replay",
           "streaming_precision.json", 0.02),
    Number("abs.preattn_other_gpus", "00_abstract.tex",
           r"Blackwell card and (\d+) to (\d+)\s+times on the others",
           port_pre_attn, "x faster", "pre-score fused attention, N=512, k=2, "
           "A100/H100/H200", "portability/*/prescore.json", 0.1),
    # introduction
    Number("intro.thirty_seven", "01_intro.tex", r"the gathered form by up to (thirty-seven) times",
           lambda: core_ratio("topk_gather", 512, 128), "x",
           "composed gathered, N=512, k=128, d=16", "attention_core_d16.json",
           0.03),
    # section 4
    Number("s4.replay_384", "04_measurement.tex",
           r"medians alike \(\$([\d.]+)\$ on both",
           lambda: crossover_replay(16, 384, 1)[1], "x best dense",
           "fused k=1, d=16, N=384, CUDA-graph replay, 3 sessions",
           "crossover_graph.json", 0.02),
    Number("s4.masked_credit", "04_measurement.tex",
           r"credits it with a \$(\d+)\\%\$ saving",
           lambda: 100 * (1 - 127 / 511), "%", "N=512, k=127", "analytic", 0.01),
    Number("s4.backward", "04_measurement.tex",
           r"gathered path is \$([\d.]+)\$ times the fused dense path on the "
           r"forward pass and \$([\d.]+)\$ times",
           lambda: (backward_ratio("forward"),
                    backward_ratio("backward_total")),
           "x", "N=512, k=128, d=16", "attention_core_d16.json", 0.02),
    Number("s4.env_share", "04_measurement.tex",
           r"take \$(\d+)\$ to \$(\d+)\\%\$ of the time", env_share_range, "%",
           "MAPPO iteration, N=6 and N=24", "training_step_timing.json", 0.02),
    Number("s4.impl_spread", "04_measurement.tex",
           r"within \$(\d+)\\%\$ of one another", impl_spread_max, "%",
           "MAPPO iteration, N=6..24 (upper bound)",
           "training_step_timing.json", 1.0),
    Number("s4.composed_vs_fused", "04_measurement.tex",
           r"costs \$([\d.]+)\$\\,ms and the fused one \$([\d.]+)\$\\,ms",
           lambda: (ms(at(_core(), "topk_gather", 512, 1)),
                    ms(at(_core(), "topk_fused_gather", 512, 1))),
           "ms", "N=512, k=1, d=16", "attention_core_d16.json", 0.02),
    Number("s4.eff_sparse", "04_measurement.tex",
           r"\$(\d+)\$ to \$(\d+)\\%\$ of the device's rate against",
           lambda: autotune_eff("sparse"), "% of peak", "d in {16,32}",
           "autotune.json", 0.03),
    Number("s4.eff_dense", "04_measurement.tex",
           r"against \$(\d+)\$ to \$(\d+)\\%\$\s+for dense",
           lambda: autotune_eff("dense"), "% of peak", "d in {16,32}",
           "autotune.json", 0.03),
    Number("s4.cfgs_dense", "04_measurement.tex",
           r"configurations against \$(\d+)\$ to \$(\d+)\$ dense ones",
           lambda: autotune_cfgs("dense"), "configs", "N>192", "autotune.json",
           0),
    Number("s4.k4_range", "04_measurement.tex",
           r"fused kernel is \$([\d.]+)\$ to \$([\d.]+)\$ times slower than dense",
           best_fused_k4_range, "x", "k=4, fp32", "precision.json", 0.03),
    Number("s4.fixed_free", "04_measurement.tex",
           r"costs \$([\d.]+)\$ to \$([\d.]+)\$ times dense attention at \$512\$",
           lambda: (fixed("fixed_set", 16, 512, 1), fixed("fixed_set", 16, 512, 4)),
           "x", "graph, d=16, N=512, k in {1,4}", "fixed_set.json", 0.1),
    Number("s4.fixed_stsel", "04_measurement.tex",
           r"same peers,\s*costs \$([\d.]+)\$ to \$([\d.]+)\$ times",
           lambda: (fixed("fused_stsel", 16, 512, 1),
                    fixed("fused_stsel", 16, 512, 4)),
           "x", "graph, d=16, N=512, k in {1,4}", "fixed_set.json", 0.02),
    Number("s4.fixed_share", "04_measurement.tex",
           r"at least \$(\d+)\\%\$ of the\s*fused kernel's device time",
           lambda: 100 * fixed_share_min(), "%", "as abs.find_share",
           "fixed_set.json", 0.05),
    Number("s4.find_set", "04_measurement.tex",
           r"costs \$(\d+)\$ to \$(\d+)\$ times dense attention on its own",
           find_set_range, "x", "graph, d=16, N>=192", "fixed_set.json", 0.05),
    Number("s4.replicate", "04_measurement.tex",
           r"reproduce it \(\$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$ against "
           r"\$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$",
           lambda: (fixed("fused_stsel", 16, 512, 1),
                    fixed("fused_stsel", 16, 512, 2),
                    fixed("fused_stsel", 16, 512, 4),
                    core_ratio("topk_fused_gather", 512, 1),
                    core_ratio("topk_fused_gather", 512, 2),
                    core_ratio("topk_fused_gather", 512, 4)),
           "x", "d=16, N=512, k in {1,2,4}",
           "fixed_set.json; attention_core_d16.json", 0.02),
    Number("s4.gain_tf32", "04_measurement.tex",
           r"\$([\d.]+)\$ to \$([\d.]+)\$ times faster in\s*TF32",
           lambda: dense_gain("tf32"), "x", "best dense fp32/tf32",
           "precision.json", 0.03),
    Number("s4.gain_bf16", "04_measurement.tex",
           r"\$([\d.]+)\$ to \$([\d.]+)\$ times faster in bf16",
           lambda: dense_gain("bf16"), "x", "best dense fp32/bf16",
           "precision.json", 0.03),
    Number("s4.k1_fp32", "04_measurement.tex",
           r"in fp32 \(\$([\d.]+)\$ to\s*\$([\d.]+)\$\)",
           lambda: prec_k1_range({"fp32"}, (192, 384, 512)), "x",
           "k=1 fp32", "precision.json", 0.03),
    Number("s4.k1_reduced_large", "04_measurement.tex",
           r"in both reduced formats \(\$([\d.]+)\$ to\s*\$([\d.]+)\$\)",
           lambda: prec_k1_range({"tf32", "bf16"}, (384, 512)), "x",
           "k=1, TF32/bf16, N>=384", "precision.json", 0.03),
    Number("s4.k1_reduced_192", "04_measurement.tex",
           r"at \$192\$ agents \(\$([\d.]+)\$ to \$([\d.]+)\$\)",
           lambda: prec_k1_range({"tf32", "bf16"}, (192,)), "x",
           "k=1, TF32/bf16, N=192", "precision.json", 0.03),
    Number("s4.bf16_median", "04_measurement.tex",
           r"ratio of medians \(\$([\d.]+)\$, bf16", bf16_median_parity, "x",
           "k=1, bf16, d=16, N=512", "precision.json", 0.02),
    Number("s4.flex_wins", "04_measurement.tex",
           r"fastest dense implementation in (nine) of the twelve",
           flex_dense_wins, "configs", "TF32/bf16", "precision.json", 0),
    Number("s4.flex_topk", "04_measurement.tex",
           r"it is \$(\d+)\$ to \$(\d+)\$ times slower than dense",
           flex_topk_range, "x", "Flex top-k k=1, all formats",
           "precision.json", 0.05),
    Number("s4.grid_recall", "04_measurement.tex",
           r"at least \$([\d.]+)\\%\$ of\s*cases", grid_recall_min, "%",
           "grid kNN, all N, k", "prescore_selectors.json", 0.0001),
    Number("s4.attn_stage", "04_measurement.tex",
           r"it\s*costs \$([\d.]+)\$ times dense attention at \$512\$ agents",
           lambda: pre_attn("prescore.json", 512, 2), "x",
           "pre-score fused attention N=512 k=2", "prescore.json", 0.05),
    Number("s4.attn_stage_medians", "04_measurement.tex",
           r"ratio of medians is \$([\d.]+)\$ in that sweep but \$([\d.]+)\$",
           lambda: (pre_attn("prescore.json", 512, 2, "med"),
                    pre_attn("prescore_selectors.json", 512, 2, "med")),
           "x", "N=512 k=2", "prescore.json; prescore_selectors.json", 0.03),
    Number("s4.attn_stage_graph", "04_measurement.tex",
           r"exact top-\$k\$ set costs \$([\d.]+)\$",
           lambda: fixed("fixed_set", 16, 512, 2), "x", "graph d=16 N=512 k=2",
           "fixed_set.json", 0.05),
    Number("s4.fuse_gain", "04_measurement.tex",
           r"worth a factor of \$(\d+)\$ to \$(\d+)\$", fuse_gain_range, "x",
           "pre-score composed/fused attention", "prescore.json", 0.05),
    Number("s4.allpairs_ms", "04_measurement.tex",
           r"from \$([\d.]+)\$ to \$([\d.]+)\$\\,ms between \$192\$ and \$512\$",
           lambda: (sel_ms("radius_knn", 192), sel_ms("radius_knn", 512)),
           "ms", "k=4", "prescore_selectors.json", 0.03),
    Number("s4.allpairs_total", "04_measurement.tex",
           r"path ends \$(\d+)\$ times slower",
           lambda: pre_total("prescore_selectors.json", "radius_knn", 512, 4),
           "x", "N=512 k=4", "prescore_selectors.json", 0.02),
    Number("s4.grid_ms", "04_measurement.tex",
           r"rises from \$([\d.]+)\$ to \$([\d.]+)\$\\,ms",
           lambda: (sel_ms("grid_knn", 192), sel_ms("grid_knn", 512)), "ms",
           "k=4", "prescore_selectors.json", 0.03),
    Number("s4.grid_total", "04_measurement.tex",
           r"keeps the path \$(\d+)\$\s*times slower",
           lambda: pre_total("prescore_selectors.json", "grid_knn", 512, 4),
           "x", "N=512 k=4", "prescore_selectors.json", 0.03),
    Number("s4.zorder_sel", "04_measurement.tex",
           r"nearly flat, \$([\d.]+)\$ to \$([\d.]+)\$\\,ms", zorder_sel_range,
           "ms", "all N and k", "prescore_selectors.json", 0.03),
    Number("s4.zorder_total", "04_measurement.tex",
           r"the path costs \$([\d.]+)\$ times dense at \$512\$\s*agents and "
           r"four retained peers, reproducing the first sweep's \$([\d.]+)\$",
           lambda: (pre_total("prescore_selectors.json", "morton_window", 512, 4),
                    pre_total("prescore.json", "morton_window", 512, 4)),
           "x", "N=512 k=4", "prescore_selectors.json; prescore.json", 0.02),
    Number("s4.flex_band", "04_measurement.tex",
           r"window costs \$([\d.]+)\$ times dense", flex_band, "x",
           "N=512 k=4", "prescore_selectors.json", 0.04),
    Number("s4.zorder_medians", "04_measurement.tex",
           r"second sweep \(\$([\d.]+)\$\) but not in the\s*first, where it is "
           r"\$([\d.]+)\$",
           lambda: (pre_total("prescore_selectors.json", "morton_window", 512, 4,
                              "med"),
                    pre_total("prescore.json", "morton_window", 512, 4, "med")),
           "x", "N=512 k=4, ratio of medians", "prescore*.json", 0.02),
    Number("s4.zorder_48", "04_measurement.tex",
           r"at \$48\$ agents the Z-order path is \$(\d+)\$ times slower",
           lambda: pre_total("prescore_selectors.json", "morton_window", 48, 4),
           "x", "N=48 k=4", "prescore_selectors.json", 0.03),
    # section 5
    Number("s5.eager_graph", "08_limitations.tex",
           r"was \$([\d.]+)\$ to \$([\d.]+)\$ times its graph-replay figure",
           eager_graph_range, "x", "fixed-set kernel, k<=4, N>=192",
           "fixed_set.json", 0.03),
    Number("s5.cfgs", "08_limitations.tex",
           r"against \$(\d+)\$ to \$(\d+)\$ for the dense kernel",
           lambda: autotune_cfgs("dense"), "configs", "N>192", "autotune.json",
           0),
]


def _round_like(printed: str, value: float) -> float:
    if "." in printed:
        return round(value, len(printed.split(".")[1]))
    return round(value)


def verify() -> tuple[list[str], list[dict]]:
    fails: list[str] = []
    records: list[dict] = []
    texts = {p.name: re.sub(r"\s+", " ", p.read_text())
             for p in SEC.glob("*.tex")}
    for n in MANIFEST:
        text = texts.get(n.file, "")
        m = re.search(n.pattern, text)
        rec = {"paper_number_id": n.id, "file": n.file, "unit": n.unit,
               "configuration": n.configuration, "source_artifact": n.source,
               "tolerance": n.tol}
        if not m:
            fails.append(f"{n.id}: pattern not found in {n.file}")
            records.append({**rec, "status": "missing"})
            continue
        got = [_num(g) for g in m.groups()]
        want = n.value()
        want = list(want) if isinstance(want, tuple) else [want]
        ok = len(got) == len(want)
        for printed, g, w in zip(m.groups(), got, want):
            wr = _round_like(printed, w) if printed not in WORDS else w
            if n.tol == 0:
                ok &= g == wr
            else:
                ok &= abs(g - wr) <= n.tol * max(abs(wr), 1e-12) + 1e-9 or \
                    abs(g - w) <= n.tol * max(abs(w), 1e-12)
        rec.update({"printed": list(m.groups()), "artifact": want,
                    "status": "ok" if ok else "MISMATCH"})
        records.append(rec)
        if not ok:
            fails.append(f"{n.id}: manuscript {m.groups()} vs artifact "
                         f"{[round(x, 4) for x in want]} ({n.source})")
    return fails, records


def main() -> int:
    fails, records = verify()
    (ROOT / "results" / "paper_numbers.json").write_text(
        json.dumps(records, indent=1, default=float))
    print(f"verified {len(records)} main-text numbers")
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
