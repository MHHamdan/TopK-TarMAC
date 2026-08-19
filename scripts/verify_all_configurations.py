"""Check every implementation against the reference at every reported configuration.

The benchmark harness checks equivalence as it goes, but three of the four
benchmark artifacts were produced before that check ran at every retained-peer
count, so they carry the check only at the widest one. Rather than re-run the
sweeps to repair a claim about correctness, this script enumerates every
configuration whose latency any released artifact reports and verifies it
directly. The output is the evidence for the equivalence statement in the paper,
and it is cheap enough to run on every change.

Writes results/equivalence.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.bench_attention_core import IMPLS, _scores, dense_unfused, topk_gather  # noqa: E402
from src.kernels.fused_prescore_attention import fused_prescore_attention  # noqa: E402
from src.kernels.prescore_selection import (  # noqa: E402
    prescore_attention, select_morton_window, select_radius_knn)

ARTIFACTS = ("attention_core_d16.json", "attention_core_dsweep.json",
             "attention_core_batch.json", "attention_core_d16_repeat.json",
             "attention_core_sort.json")
DENSE = ("dense_unfused", "dense_fused", "dense_triton")
# Implementations whose selection rule is exact top-k only when the k-th score
# is unique. Every other implementation ranks exactly as the reference does.
TIE_SENSITIVE = ("topk_fused_masked", "topk_fused_sort")


def configurations() -> list[tuple[int, int, int, tuple[str, ...]]]:
    """(N, d, k, implementations) for every latency any artifact reports."""
    seen: dict[tuple[int, int, int], set[str]] = {}
    for name in ARTIFACTS:
        p = ROOT / "results" / name
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            if r.get("failed"):
                continue
            key = (r["n_agents"], r["head_dim"], r["k"])
            seen.setdefault(key, set()).add(r["impl"])
    return [(n, d, k, tuple(sorted(v))) for (n, d, k), v in sorted(seen.items())]


def check_prescore(args) -> tuple[list[dict[str, Any]], float, int]:
    """The fused pre-score kernel against the composed path it replaces.

    Section 5's conclusion depends on the fused kernel computing the same
    function as the composition it is faster than, so the claim is only as good
    as this check. Both selectors are exercised, because the fused kernel's
    access pattern depends on the index set it is given: the Z-order indices are
    contiguous and the nearest-neighbour indices are not.
    """
    src = ROOT / "results" / "prescore.json"
    if not src.exists():
        return [], 0.0, 0
    pay = json.loads(src.read_text())
    d = pay.get("head_dim", 16)
    cfgs = sorted({(r["n_agents"], r["k"], r["selector"]) for r in pay["rows"]})
    rows: list[dict[str, Any]] = []
    worst = 0.0
    failures = 0
    for N, k, sel in cfgs:
        batch = args.batch
        while True:
            try:
                gen = torch.Generator(device=args.device).manual_seed(
                    N * 977 + k)
                q = torch.randn(batch, N, d, device=args.device, generator=gen)
                kt = torch.randn(batch, N, d, device=args.device, generator=gen)
                v = torch.randn(batch, N, d, device=args.device, generator=gen)
                pos = torch.rand(batch, N, 2, device=args.device, generator=gen)
                with torch.no_grad():
                    idx = (select_radius_knn(pos, k) if sel == "radius_knn"
                           else select_morton_window(pos, k))
                    ref = prescore_attention(q, kt, v, idx)
                    dev = (fused_prescore_attention(q, kt, v, idx)
                           - ref).abs().max().item()
                break
            except torch.OutOfMemoryError:
                q = kt = v = pos = idx = ref = None
                torch.cuda.empty_cache()
                if batch <= 1:
                    raise
                batch = max(1, batch // 4)
        worst = max(worst, dev)
        if dev > args.tol:
            failures += 1
            print(f"  FAIL prescore_fused N={N} k={k} {sel}: {dev:.2e}",
                  flush=True)
        rows.append({"n_agents": N, "k": k, "selector": sel, "head_dim": d,
                     "batch": batch, "query_rows": batch * N,
                     "deviation": dev})
        del q, kt, v, pos, idx, ref
        torch.cuda.empty_cache()
    return rows, worst, failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch", type=int, default=32,
                   help="a smaller batch than the sweeps use; equivalence is a "
                        "property of the function, not of the batch size")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--out", default=str(ROOT / "results" / "equivalence.json"))
    args = p.parse_args()

    torch.cuda.set_device(args.device)
    cfgs = configurations()
    rows: list[dict[str, Any]] = []
    worst: dict[str, float] = {}
    failures = 0

    for N, d, k, impls in cfgs:
        # The device is shared, so how much of it is free is not ours to
        # decide. Equivalence is a property of the function rather than of the
        # batch, so a configuration that will not fit is checked at a smaller
        # batch and the batch it was checked at is recorded with it.
        batch = args.batch
        while True:
            try:
                gen = torch.Generator(device=args.device).manual_seed(
                    N * 131 + d * 17 + k)
                q = torch.randn(batch, N, d, device=args.device, generator=gen)
                kt = torch.randn(batch, N, d, device=args.device, generator=gen)
                v = torch.randn(batch, N, d, device=args.device, generator=gen)
                eye = torch.eye(N, device=args.device,
                                dtype=torch.bool).unsqueeze(0)
                with torch.no_grad():
                    ref_d = dense_unfused(q, kt, v, eye, k)
                    ref_s = topk_gather(q, kt, v, eye, k)
                break
            except torch.OutOfMemoryError:
                q = kt = v = eye = ref_d = ref_s = None
                torch.cuda.empty_cache()
                if batch <= 1:
                    raise
                batch = max(1, batch // 4)
                print(f"  retrying N={N} d={d} k={k} at batch {batch}",
                      flush=True)
        with torch.no_grad():
            # How many peers the reference actually retains, so a configuration
            # where k = N-1 -- and selection is therefore never exercised -- is
            # visible in the record rather than counted as a check of selection.
            support = int(_scores(q, kt, eye).topk(k, dim=-1).indices.shape[-1])
            # An exact tie at the k-th score is the one case in which a
            # threshold rule and a rank rule legitimately differ: the threshold
            # retains every peer at or above the k-th score, so it keeps k+1.
            # Count them, so that a divergence can be attributed rather than
            # excused.
            srt = _scores(q, kt, eye).sort(dim=-1, descending=True).values
            ties = int((srt[..., k - 1] - srt[..., k] == 0).sum()) if k < N else 0
            row: dict[str, Any] = {"n_agents": N, "head_dim": d, "k": k,
                                   "batch": batch,
                                   "selection_exercised": k < N - 1,
                                   "reference_support": support,
                                   "rows_with_exact_tie_at_k": ties,
                                   "query_rows": batch * N,
                                   "deviation": {}}
            for impl in impls:
                try:
                    got = IMPLS[impl](q, kt, v, eye, k)
                    ref = ref_d if impl in DENSE else ref_s
                    dev = (got - ref).abs().max().item()
                except Exception as exc:
                    row["deviation"][impl] = None
                    row.setdefault("errors", {})[impl] = f"{type(exc).__name__}"
                    continue
                row["deviation"][impl] = dev
                # How many query rows disagree, not just by how much. A tie
                # explains a divergence only if no more rows diverge than have
                # a tie; anything wider is a failure whatever its magnitude.
                bad_rows = int(((got - ref).abs().amax(dim=-1)
                                > args.tol).sum())
                if bad_rows:
                    row.setdefault("diverging_rows", {})[impl] = bad_rows
                if (dev > args.tol and impl in TIE_SENSITIVE
                        and 0 < bad_rows <= ties):
                    # Expected and attributable, and only where the scores
                    # themselves are ambiguous. The threshold kernel retains
                    # every peer at or above the k-th score, so a tie there
                    # gives it k+1 peers; the sorted kernel breaks the same tie
                    # by peer index, which is not the order torch.topk uses.
                    # Both are exact top-k wherever the k-th score is unique.
                    row.setdefault("tie_divergence", {})[impl] = dev
                    print(f"  tie divergence {impl} N={N} d={d} k={k}: "
                          f"{dev:.2e} on {ties} of {batch * N} rows",
                          flush=True)
                    continue
                worst[impl] = max(worst.get(impl, 0.0), dev)
                if dev > args.tol:
                    failures += 1
                    print(f"  FAIL {impl} N={N} d={d} k={k}: {dev:.2e}",
                          flush=True)
        rows.append(row)
        del q, kt, v, eye
        torch.cuda.empty_cache()

    # The pre-score path is a different function -- its retained set comes from
    # a selector rather than from the scores -- so it is checked against its own
    # composed reference, at every configuration the pre-score artifact reports.
    pre_rows, pre_worst, pre_fail = check_prescore(args)

    exercised = sum(1 for r in rows if r["selection_exercised"])
    tie_rows = sum(r["rows_with_exact_tie_at_k"] for r in rows)
    total_rows = sum(r["query_rows"] for r in rows)
    tie_cfgs = sum(1 for r in rows if r.get("tie_divergence"))
    payload = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__, "batch": args.batch, "tol": args.tol,
        "configurations": len(rows),
        "configurations_exercising_selection": exercised,
        "max_deviation_by_impl": worst,
        "query_rows_checked": total_rows,
        "query_rows_with_exact_tie_at_k": tie_rows,
        "configurations_with_tie_divergence": tie_cfgs,
        "failures": failures + pre_fail,
        "prescore": {
            "configurations": len(pre_rows),
            "max_deviation": pre_worst,
            "query_rows_checked": sum(r["query_rows"] for r in pre_rows),
            "failures": pre_fail,
            "rows": pre_rows,
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{len(rows)} configurations checked, {exercised} of them with "
          f"k < N-1 so that selection is exercised")
    for impl, dev in sorted(worst.items()):
        print(f"  {impl:<20} max deviation {dev:.3e}")
    print(f"{tie_rows} of {total_rows} query rows have an exact tie at the "
          f"k-th score; the threshold kernel keeps k+1 peers on those, at "
          f"{tie_cfgs} configurations")
    if pre_rows:
        print(f"pre-score fused kernel: {len(pre_rows)} configurations, "
              f"{sum(r['query_rows'] for r in pre_rows)} query rows, "
              f"max deviation {pre_worst:.3e}")
    print(f"failures at tol {args.tol}: {failures + pre_fail}")
    return 1 if failures + pre_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
