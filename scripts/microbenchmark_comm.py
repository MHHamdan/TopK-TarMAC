"""Isolated latency + memory microbenchmark of the communication module.

The paper's motivating claim is a cost claim: aggregating over $k$ peers
instead of $N-1$ should be cheaper. This script measures it directly, with
the training loop removed so that nothing but the module is timed.

Five arms, and the distinction between them is the point:

    dense        unfused reference: materialises the (B,N,N) weight matrix
                 and contracts it with a dense (B,N,N)x(B,N,d) matmul.
    dense_sdpa   the same function through the fused SDPA kernel. This is the
                 baseline a sparse method has to beat on real hardware.
    topk_masked  top-k applied as a *mask* on the dense weights. Numerically
                 sparse, computationally dense -- the zeros are materialised
                 and the dense aggregation matmul still runs. Benchmarking
                 against only this arm measures an implementation artifact.
    topk_gather  the same function computed sparsely: gather the selected
                 values into (B,N,k,d) and contract (B*N,1,k)x(B*N,k,d).
                 Executes (N-1)/k fewer aggregation MACs. **This is the
                 variant the paper's claim is about.**
    adaptive     the learned Gumbel gate, which materialises a (B,N,N-1,N)
                 selection tensor and therefore costs O(B N^3) memory.

Protocol, stated so it can be criticised:

  * Timing uses CUDA events on-device, not host wall-clock, with a full
    `torch.cuda.synchronize()` before the event region opens.
  * `collect_info` is disabled on every module. The diagnostic entropy term
    calls `.item()`, which synchronises the device *inside* the forward pass;
    timing that measures the synchronisation, not the module.
  * `--warmup` untimed iterations, then `--reps` timed ones, then the whole
    block repeats `--runs` times. We report the within-run median/IQR *and*
    the spread of medians across runs, because GPU clocks are not locked
    (see `clocks` in the output payload).
  * Executed FLOPs are counted once per configuration with
    `torch.utils.flop_counter`, so latency is always reported next to the
    arithmetic that was actually executed.

Writes ``results/microbenchmark_comm.json`` (or ``--out``).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from src.agents.mappo import AttentionComm, MAPPOConfig

ROOT = Path(__file__).resolve().parent.parent

ARMS = ("dense", "dense_sdpa", "topk_masked", "topk_gather", "adaptive")

# Arms whose FLOP count the counter cannot see: the fused SDPA kernel is not
# registered with torch.utils.flop_counter, so its attention term reads zero.
# We add the analytic term rather than reporting a number we know is wrong.
_FLOP_COUNTER_BLIND = ("dense_sdpa",)


def _arm_kwargs(arm: str, n: int, k: int) -> dict[str, Any]:
    if arm == "dense":
        return {"attn_mode": "dense"}
    if arm == "dense_sdpa":
        return {"attn_mode": "dense_sdpa"}
    if arm == "topk_masked":
        return {"attn_mode": "topk", "topk": k}
    if arm == "topk_gather":
        return {"attn_mode": "topk_gather", "topk": k}
    if arm == "adaptive":
        return {"attn_mode": "adaptive_topk"}
    raise ValueError(arm)


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def build(arm: str, n: int, k: int, msg_dim: int, in_dim: int,
          device: str, dtype: torch.dtype = torch.float32) -> AttentionComm:
    cfg = MAPPOConfig(obs_dim=18, n_actions=5, n_agents=n, hidden=in_dim,
                      use_comm=True, msg_dim=msg_dim,
                      **_arm_kwargs(arm, n, k))  # type: ignore[arg-type]
    mod = AttentionComm(cfg, in_dim=in_dim).to(device=device, dtype=dtype)
    # The diagnostic dict forces a device sync inside forward. Never time it.
    mod.collect_info = False
    return mod


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def _time_cuda(fn, reps: int, warmup: int) -> list[float]:
    """Per-iteration milliseconds measured with on-device CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()  # current device; main() pins it via set_device
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _time_cpu(fn, reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    q = statistics.quantiles(s, n=4) if len(s) >= 4 else [s[0]] * 3
    return {
        "median_ms": statistics.median(s),
        "p25_ms": q[0],
        "p75_ms": q[2],
        "min_ms": s[0],
        "mean_ms": statistics.fmean(s),
        "std_ms": statistics.pstdev(s) if len(s) > 1 else 0.0,
    }


def measure(fn, device: str, reps: int, warmup: int, runs: int) -> dict[str, Any]:
    """Repeat the whole timed block `runs` times to expose inter-run drift."""
    timer = _time_cuda if device.startswith("cuda") else _time_cpu
    per_run = [_stats(timer(fn, reps, warmup)) for _ in range(runs)]
    medians = [r["median_ms"] for r in per_run]
    best = min(per_run, key=lambda r: r["median_ms"])
    return {
        **best,
        "runs": runs,
        "run_medians_ms": medians,
        # Spread of the per-run medians: the honest uncertainty when the
        # clocks are not locked and the device is shared.
        "inter_run_spread_pct": (
            100.0 * (max(medians) - min(medians)) / min(medians)
            if min(medians) > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# FLOPs, memory, environment
# ---------------------------------------------------------------------------

def count_flops(mod: AttentionComm, x: torch.Tensor, arm: str, b: int, n: int,
                d: int) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        mod(x)
    total = counter.get_total_flops()
    if arm in _FLOP_COUNTER_BLIND:
        # Add the attention term the counter does not observe: scores and
        # aggregation, each 2*B*N*N*d.
        total += 4 * b * n * n * d
    return total


def analytic_activation_bytes(arm: str, b: int, n: int, k: int, d: int,
                              dtype_bytes: int = 4) -> int:
    """Closed form for the dominant activation tensor of each arm.

    dense / topk_masked : the (B,N,N) weight matrix          -> O(B N^2)
    dense_sdpa          : no weight matrix at all            -> O(B N d)
    topk_gather         : scores (B,N,N) + gathered (B,N,k,d)-> O(B N^2 + B N k d)
    adaptive            : the (B,N,N-1,N) selection tensor   -> O(B N^3)

    The adaptive term is the one that matters: it is cubic in N, and it is
    incurred by the arm whose entire purpose is to reduce cost.
    """
    if arm == "dense_sdpa":
        elems = b * n * d
    elif arm in ("dense", "topk_masked"):
        elems = b * n * n
    elif arm == "topk_gather":
        elems = b * n * n + b * n * k * d
    elif arm == "adaptive":
        elems = b * n * max(n - 1, 1) * n
    else:
        raise ValueError(arm)
    return elems * dtype_bytes


def sample_clocks(device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return {}
    idx = device.split(":")[1] if ":" in device else "0"
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", idx, "--query-gpu=clocks.sm,clocks.max.sm,"
             "temperature.gpu,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, sm_max, temp, util, power = [f.strip() for f in out.split(",")]
        return {"sm_mhz": float(sm), "sm_max_mhz": float(sm_max),
                "temp_c": float(temp), "util_pct": float(util),
                "power_w": float(power)}
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error": str(exc)}


def try_lock_clocks(device: str) -> dict[str, Any]:
    """Attempt `nvidia-smi -lgc`. Records the failure rather than hiding it."""
    if not device.startswith("cuda"):
        return {"locked": False, "reason": "cpu"}
    idx = device.split(":")[1] if ":" in device else "0"
    r = subprocess.run(["nvidia-smi", "-i", idx, "-lgc",
                        "1200,1200"], capture_output=True, text=True)
    ok = "not have permission" not in (r.stdout + r.stderr)
    return {
        "locked": ok,
        "reason": None if ok else (r.stdout + r.stderr).strip().splitlines()[0],
    }


def sdpa_backend_report(device: str, n: int, d: int,
                        dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    """Which SDPA backend actually serves the dense_sdpa arm.

    'Confirm the dense path uses SDPA/flash, not a naive matmul' -- this
    answers it by trying each backend in isolation and recording which ones
    can execute this shape *at this dtype*. The dtype qualifier is not
    incidental: the fused flash / mem-efficient / cuDNN kernels accept only
    half and bfloat16 inputs, so an fp32 model -- which is what this codebase
    and the MARL literature it cites actually train -- cannot reach them at
    all, and "dense attention" necessarily means an unfused matmul + softmax.
    """
    if not device.startswith("cuda"):
        return {"available": [], "note": "cpu"}
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(4, 1, n, d, device=device, dtype=dtype)
    ok = []
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION),
                          ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("cudnn", SDPBackend.CUDNN_ATTENTION),
                          ("math", SDPBackend.MATH)):
        try:
            with sdpa_kernel([backend]):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            ok.append(name)
        except Exception:
            pass
    return {"available": ok}


def profile_kernels(mod: AttentionComm, x: torch.Tensor,
                    top: int = 6) -> list[dict[str, Any]]:
    """Top CUDA kernels by self time -- evidence for what actually ran."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(5):
        mod(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(10):
            mod(x)
        torch.cuda.synchronize()
    evts = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
    return [{"kernel": e.key, "self_cuda_us": round(e.self_device_time_total, 1)}
            for e in evts if e.self_device_time_total > 0][:top]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def prepare_cell(arm: str, n: int, k: int, k_frac: float, batch: int,
                 msg_dim: int, in_dim: int, device: str, exec_mode: str,
                 dtype: torch.dtype) -> dict[str, Any]:
    """Build one arm and measure everything that must be measured in isolation.

    Peak VRAM is a device-global statistic, so it can only be attributed to an
    arm when that arm is the only one resident. Latency, by contrast, must be
    measured with the arms interleaved (see `interleaved_timing`), because this
    device is shared and contention is not stationary. Hence two phases.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    row: dict[str, Any] = {
        "arm": arm, "n_agents": n, "k": k, "k_frac": k_frac, "batch": batch,
        "msg_dim": msg_dim, "exec_mode": exec_mode,
        "dtype": str(dtype).replace("torch.", ""),
        "analytic_activation_mib": round(
            analytic_activation_bytes(arm, batch, n, k, msg_dim, itemsize)
            / 2 ** 20, 3),
        # There is no device-memory statistic on CPU; the key must still
        # exist so downstream readers do not have to special-case the backend.
        "peak_vram_mib_forward": None,
        "peak_vram_mib_fwdbwd": None,
    }
    try:
        mod = build(arm, n, k, msg_dim, in_dim, device, dtype)
        x = torch.randn(batch, n, in_dim, device=device, dtype=dtype)
        row["flops_per_forward"] = count_flops(mod, x, arm, batch, n, msg_dim)

        callable_mod: Any = mod
        if exec_mode == "compile":
            callable_mod = torch.compile(mod, dynamic=False)
        elif exec_mode == "cudagraph":
            callable_mod = torch.compile(mod, mode="reduce-overhead",
                                         dynamic=False)

        # --- isolated peak memory, forward only -------------------------
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            callable_mod(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            row["peak_vram_mib_forward"] = (
                torch.cuda.max_memory_allocated(device) / 2 ** 20)

        # --- isolated peak memory, forward+backward ---------------------
        xg = x.detach().requires_grad_(True)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        out, _ = callable_mod(xg)
        out.sum().backward()
        mod.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            row["peak_vram_mib_fwdbwd"] = (
                torch.cuda.max_memory_allocated(device) / 2 ** 20)

        row["oom"] = False
        row["_mod"] = mod
        row["_callable"] = callable_mod
        row["_x"] = x
        row["_xg"] = xg
    except torch.OutOfMemoryError:
        # Not a failure to hide. The adaptive gate's activation memory grows
        # as O(B N^3); recording where it runs out is itself a result.
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        row["oom"] = True
        row["forward"] = None
        row["forward_backward"] = None
        row["peak_vram_mib_forward"] = None
        row["peak_vram_mib_fwdbwd"] = None
    return row


def interleaved_timing(rows: list[dict[str, Any]], device: str, reps: int,
                       warmup: int, runs: int, do_backward: bool) -> None:
    """Time every arm at one N with the arms interleaved rep by rep.

    Running all reps of arm A and then all reps of arm B makes the comparison
    hostage to whatever else was using the GPU during each block; on this
    shared host that produced >100% swings between identical configurations.
    Interleaving at the level of a single repetition means any interference
    burst lands on all arms at once, so the *ratios* between arms stay valid
    even when the absolute numbers drift.

    Reported per arm:
      min_ms     -- the cleanest estimate. Interference can only ever add
                    time, so the minimum over many samples is the closest
                    thing to an uncontended measurement available without
                    exclusive access to the device.
      median_ms  -- typical latency including contention.
      inter_run_spread_pct -- how far the per-run medians moved; this is the
                    honest uncertainty, and it is why min_ms is primary.
    """
    live = [r for r in rows if not r["oom"]]
    if not live:
        return
    on_cuda = device.startswith("cuda")

    # CUDA-graph replays share one memory pool, so a second arm's replay
    # invalidates the activations the first arm's backward still needs. That
    # is fundamental to graph capture, not a bug here, and it cannot be
    # reconciled with interleaving. Graph mode therefore reports forward only,
    # and says so rather than silently omitting the column.
    for r in live:
        if r["exec_mode"] == "cudagraph":
            r["forward_backward"] = None
            r["fwdbwd_unavailable_reason"] = (
                "CUDA-graph replays share a memory pool across arms; an "
                "interleaved backward reads activations a later replay has "
                "already overwritten")

    def one(r: dict[str, Any], backward: bool) -> float:
        # CUDA graphs reuse a fixed output buffer, so interleaving arms would
        # let arm B's replay overwrite arm A's output before A's backward ran.
        # `cudagraph_mark_step_begin` tells the allocator a new iteration has
        # started, which is exactly the interleaved case it exists for.
        graphed = r["exec_mode"] == "cudagraph"

        if backward:
            def body() -> None:
                if graphed:
                    torch.compiler.cudagraph_mark_step_begin()
                r["_mod"].zero_grad(set_to_none=True)
                out, _ = r["_callable"](r["_xg"])
                out.sum().backward()
        else:
            def body() -> None:
                if graphed:
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad():
                    r["_callable"](r["_x"])
        if not on_cuda:
            t0 = time.perf_counter()
            body()
            return (time.perf_counter() - t0) * 1e3
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        body()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for key, backward in (("forward", False),
                          ("forward_backward", True) if do_backward else
                          ("forward", False)):
        if key == "forward_backward" and not do_backward:
            continue
        active = [r for r in live
                  if not (backward and r["exec_mode"] == "cudagraph")]
        if not active:
            continue
        for _ in range(warmup):
            for r in active:
                one(r, backward)
        if on_cuda:
            torch.cuda.synchronize()

        per_run: dict[int, list[list[float]]] = {id(r): [] for r in active}
        for _run in range(runs):
            samples: dict[int, list[float]] = {id(r): [] for r in active}
            for _rep in range(reps):
                for r in active:            # <- interleaved here
                    samples[id(r)].append(one(r, backward))
            for r in active:
                per_run[id(r)].append(samples[id(r)])

        for r in active:
            blocks = per_run[id(r)]
            flat = [v for b in blocks for v in b]
            medians = [statistics.median(b) for b in blocks]
            st = _stats(flat)
            st["runs"] = runs
            st["run_medians_ms"] = medians
            st["inter_run_spread_pct"] = (
                100.0 * (max(medians) - min(medians)) / min(medians)
                if min(medians) > 0 else 0.0)
            r[key] = st
        if do_backward and key == "forward_backward":
            break


def release(rows: list[dict[str, Any]], device: str) -> None:
    for r in rows:
        for handle in ("_mod", "_callable", "_x", "_xg"):
            r.pop(handle, None)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-list", nargs="+", type=int,
                   default=[6, 12, 24, 48, 96, 192, 384, 512])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--exec-modes", nargs="+", default=["eager"],
                   choices=["eager", "compile", "cudagraph"])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--msg-dim", type=int, default=16)
    p.add_argument("--in-dim", type=int, default=64)
    p.add_argument("--k-frac", nargs="+", type=float, default=[0.25],
                   help="k = max(1, floor(k_frac * (N-1))). Multiple values "
                        "sweep the sparsity level: the gather path's advantage "
                        "depends on k, so a single k cannot settle the claim.")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--runs", type=int, default=3,
                   help="repeats of the whole timed block, for inter-run spread")
    p.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    p.add_argument("--no-backward", action="store_true")
    p.add_argument("--profile-kernels", action="store_true",
                   help="dump top CUDA kernels per arm at the largest N")
    p.add_argument("--out", default=str(ROOT / "results" / "microbenchmark_comm.json"))
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; refusing to report GPU latency from CPU")
        return 1
    if device.startswith("cuda"):
        # Pin the current device so bare `torch.cuda.synchronize()` calls wait
        # on the queue the work was actually submitted to. Without this, timing
        # anything on a non-default device measures the event resolution floor.
        torch.cuda.set_device(device)

    free_b = total_b = None
    if device.startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info(device)

    lock = try_lock_clocks(device)
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    for n in args.n_list:
        # Every arm at this N is built first, then timed interleaved, so that
        # the arms are compared against each other under the *same* contention.
        group: list[dict[str, Any]] = []
        for k_frac in args.k_frac:
            k = max(1, int(k_frac * (n - 1)))
            for arm in args.arms:
                # Arms with no k need not repeat across the k sweep.
                if (arm in ("dense", "dense_sdpa", "adaptive")
                        and k_frac != args.k_frac[0]):
                    continue
                for exec_mode in args.exec_modes:
                    group.append(prepare_cell(
                        arm, n, k, k_frac, args.batch, args.msg_dim,
                        args.in_dim, device, exec_mode, DTYPES[args.dtype]))

        interleaved_timing(group, device, args.reps, args.warmup, args.runs,
                           not args.no_backward)
        for row in group:
            row["clocks"] = sample_clocks(device)
            if row["oom"]:
                print(f"N={n:>4} k={row['k']:<4} {row['arm']:<12} "
                      f"{row['exec_mode']:<10} OUT OF MEMORY "
                      f"(analytic {row['analytic_activation_mib']:.0f} MiB)",
                      flush=True)
            else:
                f = row["forward"]
                print(f"N={n:>4} k={row['k']:<4} {row['arm']:<12} "
                      f"{row['exec_mode']:<10} "
                      f"min {f['min_ms']:8.4f} ms  med {f['median_ms']:8.4f} ms "
                      f"(+/-{f['inter_run_spread_pct']:5.1f}%)  "
                      f"{row['flops_per_forward'] / 1e6:9.2f} MFLOP  "
                      f"vram {row['peak_vram_mib_forward'] or 0:8.1f} MiB",
                      flush=True)
        release(group, device)
        rows.extend(group)

    payload: dict[str, Any] = {
        "device": device,
        "device_name": (torch.cuda.get_device_name(device)
                        if device.startswith("cuda") else platform.processor()
                        or "cpu"),
        "capability": (list(torch.cuda.get_device_capability(device))
                       if device.startswith("cuda") else None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch": args.batch,
        "dtype": args.dtype,
        "msg_dim": args.msg_dim,
        "in_dim": args.in_dim,
        "k_frac": args.k_frac,
        "reps": args.reps,
        "warmup": args.warmup,
        "runs": args.runs,
        "exec_modes": args.exec_modes,
        "timing": "cuda_events" if device.startswith("cuda") else "perf_counter",
        "collect_info_disabled": True,
        "clocks_locked": lock,
        "clocks_at_start": sample_clocks(device),
        "device_free_mib_at_start": (free_b / 2 ** 20) if free_b else None,
        "device_total_mib": (total_b / 2 ** 20) if total_b else None,
        "sdpa_backends": sdpa_backend_report(device, max(args.n_list),
                                             args.msg_dim,
                                             DTYPES[args.dtype]),
        "elapsed_s": round(time.time() - t0, 1),
        "rows": rows,
    }

    if args.profile_kernels and device.startswith("cuda"):
        n = max(args.n_list)
        k = max(1, int(args.k_frac[0] * (n - 1)))
        prof: dict[str, Any] = {}
        for arm in args.arms:
            try:
                mod = build(arm, n, k, args.msg_dim, args.in_dim, device,
                            DTYPES[args.dtype])
                x = torch.randn(args.batch, n, args.in_dim, device=device,
                                dtype=DTYPES[args.dtype])
                prof[arm] = profile_kernels(mod, x)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                prof[arm] = [{"kernel": "OOM", "self_cuda_us": 0.0}]
        payload["kernels_at_max_n"] = {"n_agents": n, "arms": prof}

    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}  ({payload['elapsed_s']} s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
