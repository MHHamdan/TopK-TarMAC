"""Tile search for the streaming top-k kernel, as autotune_kernels.py does for
the resident-row kernels.

Every (merge scheme, BLOCK_M, BLOCK_N, num_warps, num_stages) in the grid below is timed
(minimum of ``--reps`` CUDA-event samples, after warm-up) at each (d, N, k),
after checking it against the composed reference to 1e-4. Above N=512 the dense
kernel is searched too, over autotune_kernels.py's own dense space, because
results/autotune.json stops at N=512 and the streaming kernel is benchmarked
beyond it; without that the dense baseline there would be an untuned default.

Writes results/autotune_streaming.json (rows of kind "streaming" and "dense",
in the same layout as results/autotune.json), which
``streaming_topk_attention`` reads for its default tiling. Rows from an existing
output file are kept unless the same (kind, d, N, k) is re-searched, so the
search can be run in short pieces.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.autotune_kernels import _time, dense_variants, search  # noqa: E402
from src.kernels import fused_topk_attention as fta  # noqa: E402
from src.kernels import streaming_topk_attention as sta  # noqa: E402


# Merge schemes searched, as (name, BITONIC, EARLY_EXIT): bitonic top-k of the
# tile then of (buffer ++ tile top-k); k max-and-replace rounds; and rounds that
# stop once no row in the program can still improve its set.
MERGES = (("bitonic", True, False), ("rounds", False, False),
          ("rounds_exit", False, True))


def streaming_variants(q, k, v, topk, precision="ieee"):
    out = {}
    bnp = sta._next_pow2(q.shape[1])
    for (mode, bit, early), bm, bn, warps, stages in itertools.product(
            MERGES, (32, 64, 128), (16, 32, 64, 128), (2, 4, 8), (1, 2)):
        if bm > bnp or bn > bnp or bm * bn > 8192:
            continue
        cfg = {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": warps,
               "num_stages": stages, "BITONIC": bit,
               "EARLY_EXIT": early}
        out[f"stream_{mode}_m{bm}_n{bn}_w{warps}_s{stages}"] = (
            lambda c=cfg: sta.streaming_topk_attention(
                q, k, v, topk, block_m=c["BLOCK_M"], block_n=c["BLOCK_N"],
                num_warps=c["num_warps"], num_stages=c["num_stages"],
                bitonic=c["BITONIC"],
                early_exit=c["EARLY_EXIT"], tuned=False,
                precision=precision), cfg)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-list", nargs="+", type=int,
                   default=[192, 384, 512, 1024, 2048])
    p.add_argument("--k-list", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--d-list", nargs="+", type=int, default=[16, 32])
    p.add_argument("--dense-above", type=int, default=512,
                   help="also search dense tilings for N above this")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--mode", default="fp32", choices=["fp32", "tf32", "bf16"],
                   help="numerical format, as in scripts/bench_precision.py; "
                        "tf32/bf16 admit a tiling by the precision study's "
                        "row criterion against a float64 reference")
    p.add_argument("--ignore-admission", action="store_true",
                   help="tf32/bf16: time every tiling even where the format "
                        "changes the retained set in more than ROW_TOL of "
                        "rows (for reporting configurations the row "
                        "criterion excludes)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = str(ROOT / "results" / ("autotune_streaming.json"
                                           if args.mode == "fp32" else
                                           "autotune_streaming_precision.json"))
    if args.mode != "fp32":
        return precision_main(args)

    device = args.device
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out_path = Path(args.out)
    rows = json.loads(out_path.read_text())["rows"] if out_path.exists() else []

    def put(row):
        key = (row["kind"], row["head_dim"], row["n_agents"], row.get("k"),
               row["batch"])
        rows[:] = [r for r in rows if (r["kind"], r["head_dim"], r["n_agents"],
                                       r.get("k"), r["batch"]) != key]
        rows.append(row)
        out_path.write_text(json.dumps(
            {"device_name": torch.cuda.get_device_name(device),
             "torch": torch.__version__, "triton": triton.__version__,
             "reps": args.reps, "rows": rows}, indent=1) + "\n")

    t0 = time.time()
    for d in args.d_list:
        for N in args.n_list:
            B = args.batch
            g = torch.Generator(device=device).manual_seed(7)
            q = torch.randn(B, N, d, device=device, generator=g)
            k = torch.randn(B, N, d, device=device, generator=g)
            v = torch.randn(B, N, d, device=device, generator=g)
            if N > args.dense_above:
                eye = torch.eye(N, device=device, dtype=torch.bool)
                s = (q @ k.transpose(-1, -2) / d ** 0.5).masked_fill(
                    eye, float("-inf"))
                ref_dense = F.softmax(s, dim=-1) @ v
                del s
                dv = search(dense_variants(q, k, v, device), ref_dense,
                            device, args.reps)
                ok = sorted([r for r in dv if r["ms"]], key=lambda r: r["ms"])
                print(f"d={d:<3} N={N:<5} dense best {ok[0]['ms']:.4f} ms "
                      f"({ok[0]['name']})", flush=True)
                put({"kind": "dense", "head_dim": d, "n_agents": N,
                     "batch": B, "best": ok[0], "n_configs": len(dv),
                     "n_ran": len(ok), "all": ok[:5]})
            for topk in args.k_list:
                ref = fta.reference_topk_attention(q, k, v, topk)
                sv = search(streaming_variants(q, k, v, topk), ref, device,
                            args.reps)
                ok = sorted([r for r in sv if r["ms"]], key=lambda r: r["ms"])
                bad = [r for r in sv if not r["ms"]]
                print(f"d={d:<3} N={N:<5} k={topk}  streaming best "
                      f"{ok[0]['ms']:.4f} ms ({ok[0]['name']}) "
                      f"[{len(ok)}/{len(sv)} ran]", flush=True)
                put({"kind": "streaming", "head_dim": d, "n_agents": N,
                     "batch": B, "k": topk, "best": ok[0],
                     "n_configs": len(sv), "n_ran": len(ok), "all": ok[:5],
                     "failed": [{"name": r["name"], "error": r["error"]}
                                for r in bad][:10]})
            del q, k, v
            torch.cuda.empty_cache()
    print(f"wrote {out_path} in {time.time() - t0:.0f}s")
    return 0


def precision_main(args) -> int:
    """Per-format search, mirroring scripts/bench_precision.py: inputs built in
    the format (bf16 storage, or fp32 storage with TF32 products), seeded as
    there, and a tiling admitted when at most ROW_TOL of rows deviate from the
    float64 top-k reference by more than the format's tolerance."""
    from scripts import bench_precision as bp

    device = args.device
    torch.cuda.set_device(device)
    dtype, prec, tf32 = bp.MODES[args.mode]
    bp.set_tf32(tf32)
    tol = {"tf32": 2e-2, "bf16": 5e-2}[args.mode]
    out_path = Path(args.out)
    rows = json.loads(out_path.read_text())["rows"] if out_path.exists() else []

    def put(row):
        key = (row["mode"], row["head_dim"], row["n_agents"], row["k"],
               row["batch"])
        rows[:] = [r for r in rows if (r["mode"], r["head_dim"],
                                       r["n_agents"], r["k"], r["batch"])
                   != key]
        rows.append(row)
        out_path.write_text(json.dumps(
            {"device_name": torch.cuda.get_device_name(device),
             "torch": torch.__version__, "triton": triton.__version__,
             "reps": args.reps, "row_tol": bp.ROW_TOL, "rows": rows},
            indent=1) + "\n")

    t0 = time.time()
    for d in args.d_list:
        for N in args.n_list:
            B = args.batch
            gen = torch.Generator(device=device).manual_seed(77 + N + d)
            q, k, v = (torch.randn(B, N, d, device=device, generator=gen)
                       .to(dtype) for _ in range(3))
            ks = [kk for kk in args.k_list if kk < N]
            refs = bp.references(q, k, v, ks)
            for topk in ks:
                res = []
                for name, (fn, cfg) in streaming_variants(
                        q, k, v, topk, prec).items():
                    try:
                        dev, frac = bp.disagreement(fn(), refs[topk], tol)
                        if not frac <= bp.ROW_TOL and not args.ignore_admission:
                            res.append({"name": name, "cfg": cfg, "ms": None,
                                        "rows_differing": frac})
                            continue
                        res.append({"name": name, "cfg": cfg,
                                    "ms": _time(fn, device, args.reps),
                                    "deviation": dev, "rows_differing": frac})
                    except Exception as exc:
                        res.append({"name": name, "cfg": cfg, "ms": None,
                                    "error": f"{type(exc).__name__}: {exc}"[:120]})
                ok = sorted([r for r in res if r["ms"]], key=lambda r: r["ms"])
                best = ok[0] if ok else None
                print(f"{args.mode} d={d:<3} N={N:<5} k={topk}  streaming best "
                      + (f"{best['ms']:.4f} ms ({best['name']}, rows "
                         f"{best['rows_differing']:.2e})" if best else "NONE")
                      + f" [{len(ok)}/{len(res)} admitted]", flush=True)
                put({"kind": "streaming", "mode": args.mode, "head_dim": d,
                     "admission_ignored": args.ignore_admission,
                     "n_agents": N, "batch": B, "k": topk, "best": best,
                     "n_configs": len(res), "n_admitted": len(ok),
                     "all": ok[:5],
                     "min_rows_differing": min(
                         (r["rows_differing"] for r in res
                          if "rows_differing" in r), default=None)})
            del q, k, v, refs
            torch.cuda.empty_cache()
    bp.set_tf32(False)
    print(f"wrote {out_path} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
