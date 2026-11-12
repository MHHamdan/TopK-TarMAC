"""Does streaming the keys close the gap between fused top-k and dense?

Every fused score-then-select kernel in the paper holds the query's whole score
row on chip before ranking it. src/kernels/streaming_topk_attention.py does not:
it walks the keys in tiles, as a FlashAttention-style dense kernel does, and
keeps only a k-entry candidate buffer per query. This sweep times it in the
same interleaved groups as

    dense_fused      PyTorch's fused SDPA path
    dense_tuned      the Triton dense kernel at its searched tiling
    fused_stsel      the resident-row fused score-then-select kernel (N <= 512)
    streaming        the streaming kernel at its searched tiling
    fixed_set        the fused fixed-set kernel given the exact indices free
                     (N <= 512)

with the protocol of scripts/bench_fixed_set.py: one interleaved group per
(d, N) holding every k, each group timed eagerly and as CUDA-graph replays, raw
per-repetition samples stored, ratios against the better dense path under
ratio-of-minima and paired-median estimators. Above N=512 only the dense paths
and the streaming kernel are timed: the resident-row kernel is capped at 1024
and the comparison of interest there is streaming against dense.

Equivalence is checked at every timed configuration, on the timed inputs and on
extra seeds: maximum absolute deviation from the composed reference, rows above
1e-4, and the fraction of rows whose retained index set differs from
``torch.topk``'s, with the count of rows that have an exact tie at the k-th
score.

Writes results/streaming_topk.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.bench_attention_core import (  # noqa: E402
    _scores, _tuned_dense_cfg, dense_fused, topk_fused_gather, topk_gather)
from scripts.bench_fixed_set import as_graphs, summarise, time_group  # noqa: E402
from src.kernels.fused_dense_attention import fused_dense_attention  # noqa: E402
from src.kernels.fused_prescore_attention import (  # noqa: E402
    fused_prescore_attention)
from src.kernels.streaming_topk_attention import (  # noqa: E402
    _tuned_cfg as streaming_cfg, streaming_topk_attention)

RESIDENT_MAX_N = 512


def _dense_cfg(N: int, d: int, B: int):
    """The searched dense tiling: results/autotune*.json up to N=512, and
    results/autotune_streaming.json above it."""
    cfg = _tuned_dense_cfg(N, d, B)
    if cfg is not None:
        return cfg
    f = ROOT / "results" / "autotune_streaming.json"
    if f.exists():
        for r in json.loads(f.read_text())["rows"]:
            if (r["kind"] == "dense" and r["n_agents"] == N
                    and r["head_dim"] == d and r["batch"] == B
                    and "BLOCK_M" in r["best"]["cfg"]):
                return r["best"]["cfg"]
    return None


def dense_tuned(q, k, v, eye, topk):
    B, N, d = q.shape
    cfg = _dense_cfg(N, d, B)
    if cfg is None:
        return fused_dense_attention(q, k, v)
    return fused_dense_attention(q, k, v, block_m=cfg["BLOCK_M"],
                                 block_kn=cfg["BLOCK_KN"],
                                 num_warps=cfg["num_warps"],
                                 num_stages=cfg["num_stages"])


def streaming(q, k, v, eye, topk):
    return streaming_topk_attention(q, k, v, topk)


DENSE_IMPLS: dict[str, Callable] = {"dense_fused": dense_fused,
                                    "dense_tuned": dense_tuned}


def _err(got, ref, scale):
    e = (got - ref).abs()
    return {"max_abs": e.max().item(), "mean_abs": e.mean().item(),
            "max_rel": e.max().item() / scale,
            "diverging_rows": int((e.amax(-1) > 1e-4).sum())}


def verify(q, kt, v, eye, k: int, resident: bool) -> dict[str, Any]:
    with torch.no_grad():
        ref = topk_gather(q, kt, v, eye, k)
        s = _scores(q, kt, eye)
        idx = s.topk(k, dim=-1, sorted=False).indices
        srt = s.sort(dim=-1, descending=True).values
        ties = int(((srt[..., k - 1] - srt[..., k]) == 0).sum())
        got, sidx = streaming_topk_attention(q, kt, v, k, return_indices=True)
        same = (sidx.long().sort(-1).values == idx.sort(-1).values).all(-1)
        scale = ref.abs().max().item()
        res: dict[str, Any] = {
            "rows": q.shape[0] * q.shape[1],
            "rows_with_exact_tie_at_k": ties,
            "streaming": {**_err(got, ref, scale),
                          "set_mismatch_rows": int((~same).sum()),
                          "set_mismatch_frac": float((~same).float().mean())},
        }
        if resident:
            res["fused_stsel"] = _err(topk_fused_gather(q, kt, v, eye, k),
                                      ref, scale)
            res["fixed_set"] = _err(fused_prescore_attention(q, kt, v, idx),
                                    ref, scale)
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-list", nargs="+", type=int,
                   default=[192, 384, 512, 1024, 2048])
    p.add_argument("--k-list", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--d-list", nargs="+", type=int, default=[16, 32])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--extra-seeds", type=int, default=3,
                   help="additional input draws checked for equivalence")
    p.add_argument("--modes", nargs="+", default=["eager", "graph"],
                   choices=["eager", "graph"])
    p.add_argument("--out", default=str(ROOT / "results" / "streaming_topk.json"))
    p.add_argument("--merge", action="store_true",
                   help="keep groups already in --out for other (d, N)")
    args = p.parse_args()

    device, B = args.device, args.batch
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(0)
    t0 = time.time()
    groups: list[dict[str, Any]] = []
    for d in args.d_list:
        for N in args.n_list:
            resident = N <= RESIDENT_MAX_N
            gen = torch.Generator(device=device).manual_seed(1234 + N * 7 + d)
            q = torch.randn(B, N, d, device=device, generator=gen)
            kt = torch.randn(B, N, d, device=device, generator=gen)
            v = torch.randn(B, N, d, device=device, generator=gen)
            eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
            fns: dict[str, Callable[[], Any]] = {
                n: (lambda f=f: f(q, kt, v, eye, 1))
                for n, f in DENSE_IMPLS.items()}
            checks: dict[int, Any] = {}
            ks = [k for k in args.k_list if k < N]
            for k in ks:
                fns[f"streaming@{k}"] = (
                    lambda k=k: streaming(q, kt, v, eye, k))
                if resident:
                    fns[f"fused_stsel@{k}"] = (
                        lambda k=k: topk_fused_gather(q, kt, v, eye, k))
                    idx = _scores(q, kt, eye).topk(
                        k, dim=-1, sorted=False).indices
                    fns[f"fixed_set@{k}"] = (
                        lambda i=idx: fused_prescore_attention(q, kt, v, i))
                checks[k] = {"timed_inputs": verify(q, kt, v, eye, k,
                                                    resident)}
                extra = []
                for sd in range(args.extra_seeds):
                    g2 = torch.Generator(device=device).manual_seed(
                        99_000 + 101 * sd + N + d)
                    q2, k2, v2 = (torch.randn(B, N, d, device=device,
                                              generator=g2) for _ in range(3))
                    extra.append(verify(q2, k2, v2, eye, k, resident))
                    del q2, k2, v2
                checks[k]["extra_seeds"] = extra
                c = checks[k]["timed_inputs"]["streaming"]
                print(f"verify d={d} N={N} k={k}: max_abs {c['max_abs']:.1e} "
                      f"set_mismatch {c['set_mismatch_rows']} + "
                      f"{sum(e['streaming']['set_mismatch_rows'] for e in extra)}"
                      f" (extra seeds), ties "
                      f"{checks[k]['timed_inputs']['rows_with_exact_tie_at_k']}",
                      flush=True)
            for f in fns.values():  # compile every kernel outside the timing
                with torch.no_grad():
                    f()
            torch.cuda.synchronize(device)
            for mode in args.modes:
                timed = fns if mode == "eager" else as_graphs(fns, device)
                raw = time_group(timed, device, args.reps, args.warmup,
                                 args.runs)
                for k in ks:
                    sub = {n: raw[n] for n in DENSE_IMPLS}
                    sub.update({n.split("@")[0]: raw[n] for n in raw
                                if n.endswith(f"@{k}")})
                    g = summarise(sub, args.n_boot, rng)
                    g.update({"mode": mode, "n_agents": N, "k": k,
                              "head_dim": d, "batch": B, "dtype": "float32",
                              "streaming_cfg": streaming_cfg(d, N, k, B),
                              "dense_tuned_cfg": _dense_cfg(N, d, B),
                              "equivalence": checks[k], "raw_ms": sub})
                    groups.append(g)
                    r = g["ratios"]
                    print(f"{mode:<5} d={d:<3} N={N:<5} k={k:<2} best_dense="
                          f"{g['best_dense']:<11} "
                          + " ".join(f"{n} {r[n]['ratio_of_minima']:.2f}/"
                                     f"{r[n]['paired_median']:.2f}"
                                     for n in ("streaming", "fused_stsel",
                                               "fixed_set") if n in r),
                          flush=True)
                del timed, raw
            del q, kt, v, fns
            torch.cuda.empty_cache()

    out = Path(args.out)
    if args.merge and out.exists():
        done = {(g["head_dim"], g["n_agents"]) for g in groups}
        groups = [g for g in json.loads(out.read_text())["groups"]
                  if (g["head_dim"], g["n_agents"]) not in done] + groups
    meta = {
        "script": "scripts/bench_streaming_topk.py", "args": vars(args),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "elapsed_s_last_invocation": time.time() - t0,
        "estimators": "ratio_of_minima is primary, as elsewhere in the paper; "
                      "paired_median pairs repetition i with the best dense "
                      "path's repetition i in the same interleaved round",
    }
    out.write_text(json.dumps({"meta": meta, "groups": groups}, indent=1))
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
