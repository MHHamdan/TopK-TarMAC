# Drift control

The identical sweep, re-run after the original. Agreement is reported both on absolute `min_ms` and on the arm ratios the paper's claims are actually stated in.

## Summary

| Comparison | base | repeat | median abs diff (latency) | max abs diff (latency) | median abs diff (**ratio**) | max abs diff (**ratio**) |
|---|---|---|---|---|---|---|
| same device, ~21 h later, under concurrent training load | `cuda:0` | `cuda:0` | 41.3% | 80.6% | **25.9%** | **56.4%** |
| second device (same model), idle | `cuda:0` | `cuda:1` | 1.9% | 6.5% | **1.0%** | **3.2%** |

### same device, ~21 h later, under concurrent training load

gathered/dense ratio, original vs repeat:

| N | base ratio | repeat ratio | diff |
|---|---|---|---|
| 6 | 1.17x | 1.42x | +20.8% |
| 12 | 1.21x | 1.58x | +30.9% |
| 24 | 1.22x | 1.76x | +44.1% |
| 48 | 1.23x | 1.92x | +56.4% |
| 96 | 1.81x | 2.49x | +38.1% |
| 192 | 3.75x | 4.23x | +12.6% |
| 384 | 2.74x | 2.77x | +1.3% |
| 512 | 3.53x | 3.57x | +1.1% |

### second device (same model), idle

gathered/dense ratio, original vs repeat:

| N | base ratio | repeat ratio | diff |
|---|---|---|---|
| 6 | 1.17x | 1.14x | -3.1% |
| 12 | 1.21x | 1.19x | -1.6% |
| 24 | 1.22x | 1.20x | -2.2% |
| 48 | 1.23x | 1.19x | -3.2% |
| 96 | 1.81x | 1.80x | -0.5% |
| 192 | 3.75x | 3.77x | +0.4% |
| 384 | 2.74x | 2.74x | +0.3% |
| 512 | 3.53x | 3.52x | -0.4% |

## Reading

**No condition produces a crossover.** The smallest gathered/dense ratio observed anywhere across all runs, devices and load conditions is **1.14x** -- still above 1.0, i.e. gathering is slower in every cell of every repeat.

On an **idle second device of the same model**, the ratios reproduce to within 3.2% at every N (median 1.0%). The measurement is therefore a property of the algorithm and the architecture, not of one physical card or one run.

On the **same device under concurrent training load**, agreement is much worse -- ratios move by up to 56.4%. Two things are worth stating plainly about this. First, the drift is concentrated at small and moderate N, where the kernels are launch-bound and therefore most sensitive to competition for the device; the largest cells (N=384, 512) move by about 1%. Second, **every drift is in the direction that makes the sparse path look worse**, so contention cannot be masking a crossover -- it can only exaggerate a gap that is already there.

The operational consequence: latency numbers in this work must be taken on an unloaded device. The interleaved, minimum-over-samples protocol substantially reduces contention sensitivity but does not eliminate it, and claiming otherwise would overstate what the protocol buys.

