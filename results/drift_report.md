# Drift control

The identical sweep, re-run after the original. Agreement is reported both on absolute `min_ms` and on the arm ratios the paper's claims are actually stated in.

## Summary

| Comparison | base | repeat | median abs diff (latency) | max abs diff (latency) | median abs diff (**ratio**) | max abs diff (**ratio**) |
|---|---|---|---|---|---|---|
| same device, idle, later run | `cuda:0` | `cuda:0` | 0.7% | 2.6% | **0.4%** | **1.4%** |
| same device, concurrent GPU load | `cuda:0` | `cuda:0` | 67.9% | 114.2% | **33.7%** | **115.9%** |

### same device, idle, later run

gathered/dense ratio, original vs repeat:

| N | base ratio | repeat ratio | diff |
|---|---|---|---|
| 6 | 1.15x | 1.15x | -0.4% |
| 12 | 1.16x | 1.16x | -0.3% |
| 24 | 1.17x | 1.18x | +0.4% |
| 48 | 1.15x | 1.16x | +0.5% |
| 96 | 1.61x | 1.63x | +1.4% |
| 192 | 2.90x | 2.93x | +1.3% |
| 384 | 2.46x | 2.45x | -0.2% |
| 512 | 3.28x | 3.29x | +0.1% |

### same device, concurrent GPU load

gathered/dense ratio, original vs repeat:

| N | base ratio | repeat ratio | diff |
|---|---|---|---|
| 6 | 1.15x | 1.38x | +19.3% |
| 12 | 1.16x | 1.39x | +19.3% |
| 24 | 1.17x | 1.48x | +26.5% |
| 48 | 1.15x | 1.75x | +51.7% |
| 96 | 1.61x | 2.27x | +40.8% |
| 192 | 2.90x | 3.25x | +12.3% |
| 384 | 2.46x | 4.64x | +89.0% |
| 512 | 3.28x | 7.09x | +115.9% |

## Reading

The smallest gathered/dense ratio in **these drift artifacts** (same device, idle, later run, same device, concurrent GPU load) is **1.15x**, still above 1.0. This is a statement about the runs compared here only: it is not a minimum over every run in the project, and in particular it does not cover other devices, precisions or execution modes, which are reported separately.

On **same device, idle, later run**, ratios move by a median of 0.4% and at most 1.4% at any N; 5 of 8 cells move against the sparse path and 3 of 8 in its favour.

On **same device, concurrent GPU load**, ratios move by a median of 33.7% and at most 115.9% at any N; 8 of 8 cells move against the sparse path and 0 of 8 in its favour.

Drift under load is concentrated at small and moderate N, where kernels are launch-bound and therefore most sensitive to competition for the device. The directional result (8 of 8 cells adverse) comes from a single contention pattern and should be read as a margin argument, not an impossibility proof: no condition measured here moves any cell within 15% of a crossover, but an interferer that penalised the dense path more than the launch-bound sparse one is not excluded by these data.

The operational consequence: latency numbers in this work must be taken on an unloaded device. The interleaved, minimum-over-samples protocol substantially reduces contention sensitivity but does not eliminate it, and claiming otherwise would overstate what the protocol buys.

