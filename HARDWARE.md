# HARDWARE

Captured 2026-05-19.

## GPUs

4× NVIDIA GeForce RTX 2080 Ti, 11264 MiB each, driver 575.57.08, CUDA 12.9.

Per-GPU memory at detection time (shared with other processes — not exclusive):

| GPU | Used (MiB) | Free (MiB) |
|-----|------------|------------|
| 0   | 7246       | ~4018      |
| 1   | 6548       | ~4716      |
| 2   | 5140       | ~6124      |
| 3   | 6328       | ~4936      |

Implication: each device has only ~4–6 GB of free VRAM under current host load. Plan for
small models and small batch sizes. Confirm available memory just before each training
launch.

## CPU / RAM

- 187 GiB total system RAM, ~147 GiB available at detection.
- 8 GiB swap (1.4 GiB used).

## Disk

- Mount `/home` (NFS, ZFS-backed): 72 TB total, 24 TB free. Within project budget.

## Software

- Python 3.11.14 (system / conda).
- `uv` (preferred) and `pip` available.
- `git` 2.34.1.
- No `wandb` API key assumed — will run wandb in offline mode or fall back to tensorboard.

## Compute budget

- 48 GPU-hours total across the run (target).
- 200 GB disk (hard cap; check after each download).
