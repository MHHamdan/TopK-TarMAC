"""How much of a small measurement is the launch itself.

Below roughly a hundred agents the fastest configurations in this study complete
in a few tens of microseconds, which is close enough to the fixed cost of
issuing a kernel that a comparison there is partly a comparison of launch
overheads. This probe measures that fixed cost the way the sweeps measure
everything else -- CUDA events around a single call, minimum over samples -- so
the manuscript can say how close the small agent counts come to it instead of
asserting that they do.

Writes results/launch_overhead.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent.parent


@triton.jit
def _noop(X):
    tl.store(X + tl.program_id(0), 1.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--out", default=str(ROOT / "results" / "launch_overhead.json"))
    args = p.parse_args()
    torch.cuda.set_device(args.device)
    x = torch.zeros(1024, device=args.device)

    def once() -> float:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _noop[(1,)](x)
        e.record()
        torch.cuda.synchronize(args.device)
        return s.elapsed_time(e)

    for _ in range(args.warmup):
        once()
    v = sorted(once() for _ in range(args.reps))
    out = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "reps": args.reps,
        "warmup": args.warmup,
        "timing": "cuda_events_single_call",
        "empty_kernel_min_ms": v[0],
        "empty_kernel_median_ms": statistics.median(v),
        "empty_kernel_p90_ms": v[int(0.9 * len(v))],
        "note": "one program writing one float; an estimate of the fixed "
                "per-call cost as this harness measures it, not of the "
                "driver's launch latency in isolation",
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
