# HARDWARE

Captured 2026-08-18. Supersedes the 2026-08-05 capture, which described a
PyTorch build that could not execute on this hardware and quoted a lockfile hash
that no longer matches the file it names. Both problems were resolved by the
environment upgrade recorded below; the stale text is removed rather than
annotated, because a reproducibility document that describes an environment
nobody uses is worse than no document.

## GPUs

2 × **NVIDIA RTX PRO 6000 Blackwell Server Edition**, 97 887 MiB each,
driver 610.43.02, **compute capability 12.0 (sm_120)**.

⚠ **These devices are shared with other users' processes and are not
exclusively available.** Over the revision campaign they carried between 30 and
80 GiB of other tenants' allocations and near-continuous utilisation. Three
consequences, all of which the manuscript states:

- Absolute latencies are not reproducible to better than the drift reported in
  the paper's appendix. Only ratios between implementations timed in the same
  interleaved sweep should be compared.
- Peak arithmetic and peak bandwidth measured on these cards are **lower bounds**
  on what the hardware can do, so every "fraction of the machine limit" figure
  derived from them is an upper bound.
- The account lacks the privilege to lock clocks (`nvidia-smi -lgc` is refused),
  which is why the harness interleaves implementations, reports the minimum over
  samples, and publishes the spread of per-run medians.

Confirm free memory and utilisation immediately before each launch. A
configuration that builds successfully can still run out of memory mid-sweep as
another tenant grows; the benchmark harness catches that, records the
configuration as failed, and continues.

## CPU / RAM

- 2 × Intel Xeon 6507P, 32 logical cores (2 threads/core).
- 503 GiB system RAM.
- Measured fp32 matrix-multiply rate 1.77 TFLOP/s; measured copy bandwidth
  220 GB/s cache-resident and 63 GB/s from memory, giving ridge points of 8.1
  and 28.2 FLOP/byte. **An earlier version of the bandwidth probe allocated its
  source buffer without writing it**, so it read the shared zero page and
  reported roughly 1300 GB/s. Every CPU ridge point derived from that figure was
  wrong. See `scripts/roofline.py`.

## Disk

- `/home` (NFS, ZFS-backed): 72 TB total.

## Software

- Python 3.11.14 (`.venv`, managed with `uv`).
- Pinned package set: `configs/environment.lock` (107 lines),
  sha256 `d4daf16fc40b280e41b46d0f40ee55552f3c99a50678fedb2b4babea017f2068`.
- Key versions: **torch 2.11.0+cu128** (arch list includes `sm_120`, so it runs
  on these cards), **triton 3.6.0**, numpy 2.1.3, scipy 1.17.1, pandas 3.0.3,
  matplotlib 3.10.9, pettingzoo 1.26.1, mpe2 1.1.0, gymnasium 1.0.0, pytest
  9.0.3.

## Provenance of the results

- Everything under `results/attention_core_*.json`, `stage_accounting.json`,
  `prescore.json`, `training_step_timing.json` and `roofline.json` was produced
  on the hardware and software described above, by
  `scripts/run_revision_campaign.sh`.
- `results/MASTER_LOG.csv` and the `per_run` entries of
  `results/provenance.json` record **training** runs from a superseded 4 × RTX
  2080 Ti host. They are not bit-reproducible here, and the `gpu_hours` column
  in that log is wall-clock seconds ÷ 3600 rather than measured GPU occupancy.
  The top-level hardware block of `provenance.json` describes the current host,
  so that file mixes two machines and should be read with the distinction in
  mind.
- The 60 training runs carry no per-run environment record: their
  `results/<arm>/seed_<s>/config.json` files contain only the seed, and the
  arm-level configuration lives in `configs/reduced/*.yaml`. No git revision,
  package set or hardware string was captured per run.
