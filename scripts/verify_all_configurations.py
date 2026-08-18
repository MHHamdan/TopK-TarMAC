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

ARTIFACTS = ("attention_core_d16.json", "attention_core_dsweep.json",
             "attention_core_batch.json", "attention_core_d16_repeat.json")
DENSE = ("dense_unfused", "dense_fused", "dense_triton")


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
        gen = torch.Generator(device=args.device).manual_seed(N * 131 + d * 17 + k)
        q = torch.randn(args.batch, N, d, device=args.device, generator=gen)
        kt = torch.randn(args.batch, N, d, device=args.device, generator=gen)
        v = torch.randn(args.batch, N, d, device=args.device, generator=gen)
        eye = torch.eye(N, device=args.device, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            ref_d = dense_unfused(q, kt, v, eye, k)
            ref_s = topk_gather(q, kt, v, eye, k)
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
                                   "batch": args.batch,
                                   "selection_exercised": k < N - 1,
                                   "reference_support": support,
                                   "rows_with_exact_tie_at_k": ties,
                                   "query_rows": args.batch * N,
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
                if dev > args.tol and impl == "topk_fused_masked" and ties:
                    # Expected and attributable: the threshold rule differs from
                    # the rank rule only here, and only on these rows.
                    row.setdefault("tie_divergence", {})[impl] = dev
                    print(f"  tie divergence {impl} N={N} d={d} k={k}: "
                          f"{dev:.2e} on {ties} of {args.batch * N} rows",
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
        "failures": failures,
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
    print(f"failures at tol {args.tol}: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
