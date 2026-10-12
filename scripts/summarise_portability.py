"""The manuscript's headline quantities, side by side across GPUs.

Blackwell (sm_120) is read from results/; every other device from
results/portability/<tag>/, written by scripts/run_portability_campaign.sh.
Each quantity is computed by the same function of scripts/make_revision_tables.py
that computes it for the manuscript, pointed at the device's directory, so that
no figure here is defined differently from the one it is compared with.

Writes results/portability/summary.md, paper_submission_revision/tables/
portability.tex, and the per-device streaming, fixed-set and streaming-precision
tables as tables/port_<tag>_<name>.tex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.make_revision_tables as mrt  # noqa: E402

PORT = ROOT / "results" / "portability"
LABEL = {"blackwell": "RTX PRO 6000 (sm\\_120)", "h100": "H100 SXM (sm\\_90)",
         "h200": "H200 (sm\\_90)", "a100": "A100 (sm\\_80)",
         "b200": "B200 (sm\\_100)"}


def devices() -> list[tuple[str, Path]]:
    out = [("blackwell", ROOT / "results")]
    for tag in ("a100", "h100", "h200", "b200"):
        if (PORT / tag / "streaming_topk.json").exists():
            out.append((tag, PORT / tag))
    return out


def _try(f, *a):
    try:
        return f(*a)
    except (SystemExit, KeyError, ValueError, IndexError, StopIteration):
        return None


def composed_min() -> float:
    """Composed gathered top-k over the best dense, minimum over N >= 192 and
    every k: above 1 means it never wins."""
    rows = mrt.live(mrt.load("attention_core_d16.json"))
    v = []
    for N in sorted({r["n_agents"] for r in rows if r["n_agents"] >= 192}):
        bd = mrt.best_dense(rows, N)
        for r in rows:
            if r["impl"] == "topk_gather" and r["n_agents"] == N and mrt.ms(r):
                v.append(mrt.ms(r) / bd)
    return min(v)


def composed_bwd_min() -> float | None:
    """The same for the backward pass, against the best differentiable dense."""
    rows = mrt.live(mrt.load("attention_core_d16.json"))
    v = []
    for N in sorted({r["n_agents"] for r in rows if r["n_agents"] >= 192}):
        d = [mrt.ms(r, "backward_total") for r in rows
             if r["n_agents"] == N and r["impl"].startswith("dense")]
        d = [x for x in d if x]
        if not d:
            continue
        for r in rows:
            if (r["impl"] == "topk_gather" and r["n_agents"] == N
                    and mrt.ms(r, "backward_total")):
                v.append(mrt.ms(r, "backward_total") / min(d))
    return min(v) if v else None


def stream_range(k: int, d: int | None = None) -> tuple[float, float]:
    v = [g["ratios"]["streaming"]["ratio_of_minima"]
         for g in mrt.streaming_rows() if g["k"] == k
         and (d is None or g["head_dim"] == d)]
    return min(v), max(v)


def fixed_share() -> tuple[float, float]:
    G = [g for g in mrt.load("fixed_set.json")["groups"]
         if g["mode"] == "graph" and g["n_agents"] >= 384]
    s = [1 - g["ratios"]["fixed_set"]["ratio_of_minima"]
         / g["ratios"]["fused_stsel"]["ratio_of_minima"] for g in G]
    return min(s), max(s)


def resident_k1() -> tuple[float, float]:
    """Resident fused score-then-select at k=1 over best dense, d=16, N>=192,
    under graph replay (fixed-set sweep)."""
    v = [g["ratios"]["fused_stsel"]["ratio_of_minima"]
         for g in mrt.load("fixed_set.json")["groups"]
         if g["mode"] == "graph" and g["k"] == 1 and g["n_agents"] >= 192]
    return min(v), max(v)


def precision_k1(fmt: str) -> tuple[float, float] | None:
    v = [g["ratios"]["streaming"]["ratio_of_minima"]
         for g in mrt.streaming_precision_rows()
         if g["format"] == fmt and g["k"] == 1 and not g["excluded"]]
    return (min(v), max(v)) if v else None


def small_n() -> tuple[float, float, float]:
    """Best dense device time at the smallest N (us), and the composed paths'
    range over the best dense at N <= 96 (graph replay medians)."""
    pay = mrt.load("small_n.json")
    gs = [g for g in pay["groups"] if g["batch"] == 256]
    g0 = min(gs, key=lambda g: g["n_agents"])
    bd0 = min(g0["impls"][i]["graph"]["median_ms"]
              for i in ("dense_fused", "dense_tuned"))
    m = mrt.small_n_macros()
    return 1000 * bd0, float(m["SmallNComposedLo"]), float(m["SmallNComposedHi"])


def small_n_rho() -> tuple[float, float]:
    """Median within-configuration rank correlation of device time with kernel
    count and with executed operations, N <= 96 (graph replay)."""
    m = mrt.small_n_macros()
    return float(m["SmallNRhoLaunch"]), float(m["SmallNRhoFlops"])


def prescore_zorder(k: int = 4) -> tuple[float, float] | None:
    """Z-order-window selection before scoring at the largest N and k retained
    peers (the setting Section 4 quotes): (fused attention alone, selector +
    fused attention), each over the best dense."""
    rows = mrt.load("prescore.json")["rows"]
    N = max(r["n_agents"] for r in rows)
    r = [x for x in rows if x["selector"] == "morton_window"
         and x["n_agents"] == N and x["k"] == k]
    if not r or not r[0].get("t_total_fused_ms"):
        return None
    r = r[0]
    d = r.get("t_dense_best_ms", r["t_dense_fused_ms"])
    return r["t_attention_fused_ms"] / d, r["t_total_fused_ms"] / d


def machine() -> tuple[float, float] | None:
    """Measured matmul TFLOP/s and DRAM-resident copy GB/s (scripts/roofline.py)."""
    p = mrt.RES / "roofline.json"
    if not p.exists():
        return None
    for key, m in json.loads(p.read_text())["machines"].items():
        if key.startswith("cuda"):
            return m["achievable_gflops"] / 1e3, m["dram_resident_gbytes_per_s"]
    return None


BATCH_FILES = ((32, "streaming_topk_b32.json"), (256, "streaming_topk.json"),
               (1024, "streaming_topk_b1024.json"),
               (4096, "streaming_topk_b4096.json"))


def batch_ranges(path: Path) -> dict[int, dict[int, tuple[float, float]]]:
    """Streaming over best dense per batch and k, d=16, N in 192..512, graph
    replay: {batch: {k: (min, max)}}."""
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for b, f in BATCH_FILES:
        p = path / f
        if not p.exists():
            continue
        v: dict[int, list[float]] = {}
        for g in json.loads(p.read_text())["groups"]:
            if (g["mode"] == "graph" and g["head_dim"] == 16
                    and g["batch"] == b and 192 <= g["n_agents"] <= 512):
                v.setdefault(g["k"], []).append(
                    g["ratios"]["streaming"]["ratio_of_minima"])
        out[b] = {k: (min(x), max(x)) for k, x in v.items()}
    return out


def ncu_figures(path: Path, N: int = 512, k: int = 4) -> dict | None:
    """Per-kernel counters at one shape: the composed path's DRAM traffic and
    its radix-select kernels, against dense; DRAM and SM attainment of the
    fused and streaming kernels."""
    p = path / "ncu_counters.json"
    if not p.exists():
        return None
    G = {g["impl"]: g for g in json.loads(p.read_text())["groups"]
         if g["n_agents"] == N and g["k"] == k}
    if "topk_gather" not in G or "dense_tuned" not in G:
        return None
    comp = G["topk_gather"]
    sel = [x for x in comp["kernels"] if "mbtopk" in x["kernel"]]
    fused = [x for i in ("topk_fused_gather", "streaming") if i in G
             for x in G[i]["kernels"]]
    pct = lambda m: [x[m] for x in fused]
    dense = G["dense_tuned"]["kernels"]
    byt = lambda x: (x["dram__bytes_read.sum"] or 0) + (x["dram__bytes_write.sum"] or 0)
    return {
        "radix_passes": sum("computeBlockDigitCounts" in x["kernel"] for x in sel),
        "select_bytes_share": sum(byt(x) for x in sel) / comp["total"]["dram_bytes"],
        "fused_occ_pct": (min(pct("sm__warps_active.avg.pct_of_peak_sustained_active")),
                          max(pct("sm__warps_active.avg.pct_of_peak_sustained_active"))),
        "dense_dram_pct": dense[0]["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
        "dense_sm_pct": dense[0]["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
        "fixed_over_streaming": (G["fixed_set"]["total"]["time_s"]
                                 / G["streaming"]["total"]["time_s"])
        if "fixed_set" in G and "streaming" in G else None,
        "composed_gb": comp["total"]["dram_bytes"] / 1e9,
        "dense_mb": G["dense_tuned"]["total"]["dram_bytes"] / 1e6,
        "kernels": comp["n_kernels"], "select_kernels": len(sel),
        "select_share": sum(x["gpu__time_duration.sum"] for x in sel)
        / (comp["total"]["time_s"] * 1e9),
        "fused_dram_pct": (min(pct("dram__throughput.avg.pct_of_peak_sustained_elapsed")),
                           max(pct("dram__throughput.avg.pct_of_peak_sustained_elapsed"))),
        "fused_sm_pct": (min(pct("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
                         max(pct("sm__throughput.avg.pct_of_peak_sustained_elapsed"))),
    }


def _span(vals: list[tuple[float, float]]) -> tuple[float, float]:
    return min(v[0] for v in vals), max(v[1] for v in vals)


def write_macros(raw: dict, others: list[tuple[str, Path]]) -> None:
    """Prose figures for the other-GPU appendix, pooled over every device that
    is not the Blackwell card, as tables/portability_macros.tex."""
    tags = [t for t, _ in others]
    pool = lambda key: _span([raw[t][key] for t in tags if raw[t][key]])
    m: dict[str, str] = {"PortDevices": str(len(tags))}
    for k, K in ((1, "One"), (2, "Two"), (4, "Four")):
        lo, hi = pool(f"streaming fp32 k={k} ÷ best dense")
        m[f"PortStream{K}Lo"], m[f"PortStream{K}Hi"] = f"{lo:.2f}", f"{hi:.2f}"
    for fmt, F in (("TF32", "Tf"), ("bf16", "Bf")):
        lo, hi = pool(f"streaming {fmt} k=1 ÷ best {fmt} dense")
        m[f"Port{F}OneLo"], m[f"Port{F}OneHi"] = f"{lo:.2f}", f"{hi:.2f}"
    lo, hi = pool("share of fused time spent finding the set (N≥384)")
    m["PortFindLo"], m["PortFindHi"] = f"{100 * lo:.0f}", f"{100 * hi:.0f}"
    comp = [raw[t]["composed gathered ÷ best dense, min over N≥192, all k (fwd)"]
            for t in tags]
    m["PortComposedMin"] = f"{min(c for c in comp if c):.1f}"
    zo = [raw[t]["pre-score Z-order, N=512 k=4: attend; select+attend ÷ best dense"]
          for t in tags]
    zo = [z for z in zo if z]
    m["PortZorderAttendLo"] = f"{min(z[0] for z in zo):.2f}"
    m["PortZorderAttendHi"] = f"{max(z[0] for z in zo):.2f}"
    m["PortZorderTotalLo"] = f"{min(z[1] for z in zo):.2f}"
    m["PortZorderTotalHi"] = f"{max(z[1] for z in zo):.2f}"
    bw = {t: raw[t]["measured matmul TFLOP/s (roofline probe); DRAM copy GB/s"]
          for t in tags}
    if bw.get("h100") and bw.get("h200"):
        m["PortBwRatioHTwoHundred"] = f"{bw['h200'][1] / bw['h100'][1]:.1f}"
    # Per-device detail for the statements that do not pool cleanly.
    stream_two, find, attn, rho_l, rho_f = {}, [], [], [], []
    for t, p in others:
        mrt.RES = p
        stream_two[t] = sorted(
            (g["ratios"]["streaming"]["ratio_of_minima"], g["head_dim"],
             g["n_agents"]) for g in mrt.streaming_rows() if g["k"] == 2)
        find.append(mrt.streaming_find_share())
        r = [x for x in mrt.load("prescore.json")["rows"]
             if x["selector"] == "morton_window" and x["n_agents"] == 512
             and x["k"] == 2][0]
        attn.append(r["t_dense_best_ms"] / r["t_attention_fused_ms"])
        sm = mrt.small_n_macros()
        rho_l.append(float(sm["SmallNRhoLaunch"]))
        rho_f.append(float(sm["SmallNRhoFlops"]))
    if "a100" in stream_two:
        v = [x[0] for x in stream_two["a100"]]
        m["PortStreamTwoAHundredLo"] = f"{min(v):.2f}"
        m["PortStreamTwoAHundredHi"] = f"{max(v):.2f}"
    hop = [t for t in ("h100", "h200") if t in stream_two]
    if hop:
        wins = {sum(x[0] < 1 for x in stream_two[t]) for t in hop}
        worst = [stream_two[t][-1] for t in hop]
        m["PortStreamTwoConfigs"] = str(len(stream_two[hop[0]]))
        m["PortStreamTwoHopperWins"] = "/".join(str(w) for w in sorted(wins))
        m["PortStreamTwoHopperWorst"] = f"{max(w[0] for w in worst):.2f}"
        m["PortStreamTwoHopperWorstD"] = str(worst[0][1])
        m["PortStreamTwoHopperWorstN"] = str(worst[0][2])
    prec: dict[tuple[str, int], list[float]] = {}
    excluded_bf16 = 0
    for t, p in others:
        mrt.RES = p
        for g in mrt.streaming_precision_rows():
            if g["format"] == "bf16" and g["excluded"]:
                excluded_bf16 += 1
            if not g["excluded"]:
                prec.setdefault((g["format"], g["k"]), []).append(
                    g["ratios"]["streaming"]["ratio_of_minima"])
    for (fmt, k), v in prec.items():
        F = {"tf32": "Tf", "bf16": "Bf"}[fmt]
        K = {1: "One", 2: "Two", 4: "Four"}[k]
        m[f"Port{F}{K}Lo"], m[f"Port{F}{K}Hi"] = f"{min(v):.2f}", f"{max(v):.2f}"
    m["PortBfExcluded"] = str(excluded_bf16)

    def _count(x, acc):
        """Sum set_mismatch_rows and rows over a nested retained-set record."""
        if isinstance(x, dict):
            for k, v in x.items():
                if k.endswith("mismatch_rows") and isinstance(v, (int, float)):
                    acc[0] += v
                elif k == "rows" and isinstance(v, (int, float)):
                    acc[1] += v
                else:
                    _count(v, acc)
        elif isinstance(x, list):
            for v in x:
                _count(v, acc)
        return acc

    bf = []
    for t, p in [("blackwell", ROOT / "results")] + list(others):
        mrt.RES = p
        acc = [0, 0]
        for g in mrt.load("streaming_precision.json")["groups"]:
            if g["format"] == "bf16":
                _count(g["retained_set"], acc)
        bf.append(acc)
    m["PortBfMismatchMax"] = str(int(max(a[0] for a in bf)))
    m["PortBfRowsM"] = f"{min(a[1] for a in bf) / 1e6:.0f}"
    m["PortStreamFindLo"] = f"{100 * min(f[0] for f in find):.0f}"
    m["PortStreamFindHi"] = f"{100 * max(f[1] for f in find):.0f}"
    m["PortPreAttnLo"] = f"{min(attn):.0f}"
    m["PortPreAttnHi"] = f"{max(attn):.0f}"
    m["PortSmallNRhoLaunchLo"] = f"{min(rho_l):.2f}"
    m["PortSmallNRhoLaunchHi"] = f"{max(rho_l):.2f}"
    m["PortSmallNRhoFlopsHi"] = f"{max(rho_f):.2f}"
    br = {t: batch_ranges(p) for t, p in others}
    big = [br[t][b][k] for t in tags for b in (256, 1024, 4096) for k in (1,)
           if b in br[t]]
    m["PortBatchOneLo"], m["PortBatchOneHi"] = (f"{x:.2f}" for x in _span(big))
    big = [br[t][b][2] for t in tags for b in (256, 1024, 4096) if b in br[t]]
    m["PortBatchTwoLo"], m["PortBatchTwoHi"] = (f"{x:.2f}" for x in _span(big))
    small = [br[t][32][1] for t in tags if 32 in br[t]]
    m["PortBatchSmallOneLo"], m["PortBatchSmallOneHi"] = (
        f"{x:.2f}" for x in _span(small))
    nc = [f for f in (ncu_figures(p) for _, p in others) if f]
    if nc:
        m["PortNcuDevices"] = str(len(nc))
        m["PortNcuComposedGB"] = f"{min(f['composed_gb'] for f in nc):.1f}"
        m["PortNcuDenseMB"] = f"{max(f['dense_mb'] for f in nc):.0f}"
        m["PortNcuKernels"] = str(nc[0]["kernels"])
        m["PortNcuSelectKernels"] = str(nc[0]["select_kernels"])
        m["PortNcuSelectShareLo"] = f"{100 * min(f['select_share'] for f in nc):.0f}"
        m["PortNcuSelectShareHi"] = f"{100 * max(f['select_share'] for f in nc):.0f}"
        lo, hi = _span([f["fused_dram_pct"] for f in nc])
        m["PortNcuFusedDramLo"], m["PortNcuFusedDramHi"] = f"{lo:.1f}", f"{hi:.1f}"
        lo, hi = _span([f["fused_sm_pct"] for f in nc])
        m["PortNcuFusedSmLo"], m["PortNcuFusedSmHi"] = f"{lo:.0f}", f"{hi:.0f}"
        lo, hi = _span([f["fused_occ_pct"] for f in nc])
        m["PortNcuFusedOccLo"], m["PortNcuFusedOccHi"] = f"{lo:.0f}", f"{hi:.0f}"
        m["PortNcuDenseDramHi"] = f"{max(f['dense_dram_pct'] for f in nc):.1f}"
        m["PortNcuDenseSmLo"] = f"{min(f['dense_sm_pct'] for f in nc):.0f}"
        m["PortNcuDenseSmHi"] = f"{max(f['dense_sm_pct'] for f in nc):.0f}"
        m["PortNcuRadixPasses"] = str(nc[0]["radix_passes"])
        m["PortNcuSelectBytesLo"] = f"{100 * min(f['select_bytes_share'] for f in nc):.0f}"
        m["PortNcuSelectBytesHi"] = f"{100 * max(f['select_bytes_share'] for f in nc):.0f}"
        fo = [f["fixed_over_streaming"] for f in nc if f["fixed_over_streaming"]]
        m["PortNcuFixedShareLo"] = f"{100 * min(fo):.1f}"
        m["PortNcuFixedShareHi"] = f"{100 * max(fo):.1f}"
    (mrt.OUT / "portability_macros.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items())))

    # The batch table: streaming over best dense at k = 1, 2, 4, one block of
    # rows per device, so that it fits the text width at natural size.
    short = {"a100": "A100", "h100": "H100", "h200": "H200", "b200": "B200"}
    body = []
    for i, t in enumerate(tags):
        if i:
            body.append(r"\midrule")
        for j, (b, _) in enumerate(BATCH_FILES):
            cells = [rng(br[t].get(b, {}).get(k)) for k in (1, 2, 4)]
            body.append(f"{short.get(t, t) if j == 0 else ''} & {b} & "
                        + " & ".join(cells) + r" \\")
    (mrt.OUT / "port_batch.tex").write_text(
        "\\begin{tabular}{lrccc}\n\\toprule\nGPU & $B$ & $k{=}1$ & $k{=}2$ & "
        "$k{=}4$ \\\\\n\\midrule\n" + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}%\n")



def rng(t, p=2):
    return "---" if t is None else f"{t[0]:.{p}f}--{t[1]:.{p}f}"


def main() -> int:
    devs = devices()
    cols, raw = [], {}
    for tag, path in devs:
        mrt.RES = path
        row = {
            "composed gathered ÷ best dense, min over N≥192, all k (fwd)":
                _try(composed_min),
            "composed gathered ÷ best dense, min (bwd)":
                _try(composed_bwd_min),
            "resident fused k=1 ÷ best dense (graph, N≥192)": _try(resident_k1),
            "streaming fp32 k=1 ÷ best dense": _try(stream_range, 1),
            "streaming fp32 k=2 ÷ best dense": _try(stream_range, 2),
            "streaming fp32 k=4 ÷ best dense": _try(stream_range, 4),
            "streaming fp32 k=8 ÷ best dense": _try(stream_range, 8),
            "share of fused time spent finding the set (N≥384)":
                _try(fixed_share),
            "streaming kernel: share of time finding the set (N≥384)":
                _try(mrt.streaming_find_share),
            "small N: rank correlation with kernel count; with FLOPs":
                _try(small_n_rho),
            "streaming TF32 k=1 ÷ best TF32 dense": _try(precision_k1, "tf32"),
            "streaming bf16 k=1 ÷ best bf16 dense": _try(precision_k1, "bf16"),
            "small N: best dense at N=12 (µs); composed ÷ dense at N≤96":
                _try(small_n),
            "pre-score Z-order, N=512 k=4: attend; select+attend ÷ best dense":
                _try(prescore_zorder),
            "measured matmul TFLOP/s (roofline probe); DRAM copy GB/s": _try(machine),
        }
        raw[tag] = row
        cols.append(tag)
        for name, fn in (("streaming", mrt.table_streaming),
                         ("fixed_set", mrt.table_fixed_set),
                         ("streaming_precision", mrt.table_streaming_precision)):
            if tag == "blackwell":
                continue
            body = _try(fn)
            if body:
                (mrt.OUT / f"port_{tag}_{name}.tex").write_text(body + "%\n")

    def cell(v, key):
        if v is None:
            return "---"
        if key.startswith("small N: rank"):
            return f"{v[0]:.2f}; {v[1]:.2f}"
        if "share" in key:
            return f"{100 * v[0]:.0f}--{100 * v[1]:.0f}%"
        if key.startswith("measured"):
            return f"{v[0]:.0f}; {v[1]:.0f}"
        if key.startswith("pre-score"):
            return f"{v[0]:.2f}; {v[1]:.2f}"
        if key.startswith("small N"):
            return f"{v[0]:.1f}; {v[1]:.1f}--{v[2]:.1f}"
        if isinstance(v, tuple):
            return rng(v)
        return f"{v:.2f}"

    keys = list(next(iter(raw.values())))
    md = ["| quantity | " + " | ".join(cols) + " |",
          "|---|" + "---|" * len(cols)]
    for key in keys:
        md.append(f"| {key} | " + " | ".join(cell(raw[c][key], key)
                                             for c in cols) + " |")
    PORT.mkdir(parents=True, exist_ok=True)
    (PORT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    tex_rows = [
        (r"\multicolumn{%d}{l}{\emph{Time $\div$ best dense, same format (lower is better)}} \\" % (len(cols) + 1), None),
        ("Composed, fp32, forward, best",
         "composed gathered ÷ best dense, min over N≥192, all k (fwd)"),
        ("Composed, fp32, backward, best",
         "composed gathered ÷ best dense, min (bwd)"),
        ("Resident, fp32, $k{=}1$", "resident fused k=1 ÷ best dense (graph, N≥192)"),
        ("Streaming, fp32, $k{=}1$", "streaming fp32 k=1 ÷ best dense"),
        ("Streaming, fp32, $k{=}2$", "streaming fp32 k=2 ÷ best dense"),
        ("Streaming, fp32, $k{=}4$", "streaming fp32 k=4 ÷ best dense"),
        ("Streaming, fp32, $k{=}8$", "streaming fp32 k=8 ÷ best dense"),
        ("Streaming, TF32, $k{=}1$", "streaming TF32 k=1 ÷ best TF32 dense"),
        ("Streaming, bf16, $k{=}1$", "streaming bf16 k=1 ÷ best bf16 dense"),
        ("Z-order, fp32: attend; total",
         "pre-score Z-order, N=512 k=4: attend; select+attend ÷ best dense"),
        (r"\midrule \multicolumn{%d}{l}{\emph{Where the time goes}} \\" % (len(cols) + 1), None),
        ("Resident: finding the set",
         "share of fused time spent finding the set (N≥384)"),
        ("Streaming: finding the set",
         "streaming kernel: share of time finding the set (N≥384)"),
        (r"$N{\le}96$: $\rho$ kernels; $\rho$ ops",
         "small N: rank correlation with kernel count; with FLOPs"),
    ]
    head = {"blackwell": r"\shortstack{RTX PRO 6000\\(sm\_120)}",
            "a100": r"\shortstack{A100\\(sm\_80)}",
            "h100": r"\shortstack{H100 SXM\\(sm\_90)}",
            "h200": r"\shortstack{H200\\(sm\_90)}"}
    body = []
    for lab, key in tex_rows:
        if key is None:
            body.append(lab)
            continue
        body.append(lab + " & " + " & ".join(
            cell(raw[c][key], key).replace("%", "\\%") for c in cols) + r" \\")
    tex = (r"\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n& "
           + " & ".join(head.get(c, LABEL.get(c, c)) for c in cols) + r" \\"
           + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}%\n")
    (mrt.OUT / "portability.tex").write_text(tex)
    write_macros(raw, [(t, p) for t, p in devs if t != "blackwell"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
