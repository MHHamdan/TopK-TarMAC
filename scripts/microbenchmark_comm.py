"""Isolated latency microbenchmark of the communication module.

The paper's motivating claim is a cost claim: aggregating over $k$ peers
instead of $N-1$ should be cheaper. That claim has never been measured --
only counted analytically (AUDIT.md, C4). This script measures it directly,
with the training loop removed so that nothing but the module is timed.

Protocol: warm up, then take the median of ``--reps`` timed forward (and
forward+backward) passes with an explicit ``torch.cuda.synchronize()`` on
each side. Median rather than mean, because a shared GPU produces occasional
large outliers that a mean would absorb.

A top-$k$ gather that is *slower* than a dense matmul at small $N$ is a real
result, not a bug: gather kernels have launch and indexing overhead that a
single dense GEMM does not pay. The crossover is what the figure reports.

Writes ``results/microbenchmark_comm.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from src.agents.mappo import AttentionComm, MAPPOConfig

ROOT = Path(__file__).resolve().parent.parent


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def time_module(mod: AttentionComm, x: torch.Tensor, device: str,
                backward: bool, reps: int, warmup: int) -> dict[str, float]:
    """Median/IQR latency in milliseconds for one module configuration.

    Args:
        mod: The communication block to time.
        x: Input tensor of shape ``(B, N, D)``.
        device: Torch device string, used for synchronisation.
        backward: Whether to include a backward pass in the timed region.
        reps: Timed repetitions.
        warmup: Untimed repetitions run first.

    Returns:
        Mapping with median, p25, p75, min latency in ms.
    """
    for _ in range(warmup):
        out, _ = mod(x)
        if backward:
            mod.zero_grad(set_to_none=True)
            out.sum().backward()
    _sync(device)

    samples: list[float] = []
    for _ in range(reps):
        if backward:
            mod.zero_grad(set_to_none=True)
        _sync(device)
        t0 = time.perf_counter()
        out, _ = mod(x)
        if backward:
            out.sum().backward()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    q = statistics.quantiles(samples, n=4) if len(samples) >= 4 else [samples[0]] * 3
    return {
        "median_ms": statistics.median(samples),
        "p25_ms": q[0],
        "p75_ms": q[2],
        "min_ms": samples[0],
    }


def peak_vram_mib(device: str) -> float | None:
    """Peak allocated VRAM in MiB since the last reset, or None on CPU."""
    if not device.startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated(device) / (2 ** 20)


def run(n_list: list[int], device: str, batch: int, msg_dim: int,
        reps: int, warmup: int) -> list[dict[str, Any]]:
    """Benchmark dense and top-k aggregation across agent counts."""
    rows: list[dict[str, Any]] = []
    for n in n_list:
        # k = 25% of peers, the sparsity regime the Lagrangian arm targets.
        k = max(1, (n - 1) // 4)
        arms = [
            ("dense", {"attn_mode": "dense"}),
            (f"topk_k{k}", {"attn_mode": "topk", "topk": k}),
            ("topk_k_half", {"attn_mode": "topk", "topk": max(1, (n - 1) // 2)}),
            ("adaptive_topk", {"attn_mode": "adaptive_topk"}),
        ]
        for arm, kw in arms:
            cfg = MAPPOConfig(obs_dim=18, n_actions=5, n_agents=n, hidden=64,
                              use_comm=True, msg_dim=msg_dim, **kw)  # type: ignore[arg-type]
            mod = AttentionComm(cfg, in_dim=64).to(device)
            x = torch.randn(batch, n, 64, device=device, requires_grad=True)

            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            try:
                fwd = time_module(mod, x, device, backward=False,
                                  reps=reps, warmup=warmup)
                fwdbwd = time_module(mod, x, device, backward=True,
                                     reps=reps, warmup=warmup)
            except torch.OutOfMemoryError:
                # Not a failure to hide: the adaptive gate materialises a
                # (B, N, N-1, N) selection tensor, so its activation memory
                # grows as O(B N^3) while dense attention grows as O(B N^2).
                # Recording where it runs out is itself a result.
                torch.cuda.empty_cache()
                rows.append({
                    "n_agents": n, "arm": arm, "k": kw.get("topk", n - 1),
                    "batch": batch, "msg_dim": msg_dim,
                    "forward": None, "forward_backward": None,
                    "peak_vram_mib": None, "oom": True,
                })
                print(f"N={n:>4} {arm:<14} OUT OF MEMORY at batch={batch}",
                      flush=True)
                continue

            row = {
                "n_agents": n,
                "arm": arm,
                "k": kw.get("topk", n - 1),
                "batch": batch,
                "msg_dim": msg_dim,
                "forward": fwd,
                "forward_backward": fwdbwd,
                "peak_vram_mib": peak_vram_mib(device),
                "oom": False,
            }
            rows.append(row)
            print(f"N={n:>4} {arm:<14} fwd {fwd['median_ms']:7.3f} ms   "
                  f"fwd+bwd {fwdbwd['median_ms']:7.3f} ms   "
                  f"vram {row['peak_vram_mib'] or 0:7.1f} MiB", flush=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-list", nargs="+", type=int,
                   default=[6, 12, 24, 48, 96, 192])
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--msg-dim", type=int, default=16)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", default=str(ROOT / "results" / "microbenchmark_comm.json"))
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; refusing to report GPU latency from CPU")
        return 1

    t0 = time.time()
    rows = run(args.n_list, args.device, args.batch, args.msg_dim,
               args.reps, args.warmup)
    payload = {
        "device": args.device,
        "device_name": (torch.cuda.get_device_name(args.device)
                        if args.device.startswith("cuda") else "cpu"),
        "torch": torch.__version__,
        "batch": args.batch,
        "msg_dim": args.msg_dim,
        "reps": args.reps,
        "warmup": args.warmup,
        "elapsed_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}  ({payload['elapsed_s']} s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
