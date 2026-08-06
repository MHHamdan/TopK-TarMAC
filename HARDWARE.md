# HARDWARE

Captured 2026-08-05 (supersedes the 2026-05-19 capture; the previous 4× RTX 2080 Ti
host is no longer the execution environment).

## GPUs

2 × **NVIDIA RTX PRO 6000 Blackwell Server Edition**, 97 887 MiB each
(**~191 GiB aggregate, not a single 196 GiB device**), driver 610.43.02,
**compute capability 12.0 (sm_120)**.

Memory at capture time — **these devices are shared with other users' processes and are
not exclusively available**:

| GPU | Total (MiB) | Used (MiB) | Free (MiB) | Util at capture |
|-----|-------------|------------|------------|-----------------|
| 0   | 97 887      | 59 141     | 38 150     | 100 %           |
| 1   | 97 887      | 52 340     | 44 951     | 19 %            |

Implication: plan against **~38–45 GiB free per device**, not 96 GiB, and expect
contention for SM time on GPU 0. Confirm free memory and utilisation immediately before
each launch.

## ⚠ PyTorch/GPU incompatibility (blocking)

The pinned build **cannot execute on this hardware**:

```
torch 2.5.1+cu124
torch.cuda.get_arch_list() -> ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
device capability                 -> (12, 0)   # sm_120

>>> x = torch.randn(512, 512, device="cuda"); (x @ x).sum()
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

`torch.cuda.is_available()` still returns `True`, so `src/training/train_mappo.py:155`
selects `cuda` and then fails at the first kernel launch. A cu128 PyTorch build
(≥ 2.7) is required before any GPU run. See `AUDIT.md` D-00.

## CPU / RAM

- 2 × Intel Xeon 6507P, 32 logical cores (2 threads/core).
- 503 GiB system RAM, ~467 GiB available at capture.

## Disk

- `/home` (NFS, ZFS-backed): 72 TB total, 20 TB free.

## Software

- Python 3.11.14 (`.venv`, managed with `uv`).
- Pinned package set: `configs/environment.lock` (101 packages),
  sha256 `a3d58f4f6023c622c7b566d9a9321d9be3a300d2043d1006edb53048c5802dd1`.
- Key versions: torch 2.5.1+cu124, numpy 2.1.3, scipy 1.17.1, pandas 3.0.3,
  matplotlib 3.10.9, pettingzoo 1.26.1, mpe2 1.1.0, gymnasium 1.0.0, pytest 9.0.3.
- `git` 2.34.1.

## Provenance of the existing 45 runs

All runs in `results/MASTER_LOG.csv` were produced on the **previous** 4× RTX 2080 Ti
host under torch 2.5.1+cu124/sm_75. They are **not** bit-reproducible on the current
hardware, and the `gpu_hours` column in that log is wall-clock seconds ÷ 3600, not
measured GPU occupancy (`AUDIT.md` D-06).
