# Proposal 001 — TopK-TarMAC: Sparse Targeted Communication with Adaptive k

## Research question
Can top-k attention over outgoing inter-agent messages, with k adapted on a
per-agent per-step basis, match the task return of dense (soft) attention in
cooperative MARL while reducing communication-channel FLOPs from O(N²) to
O(N·k̄) for k̄ ≪ N?

## Hypothesis
On MPE simple_spread at N ∈ {3, 6, 12, 25}, an attention-communication module
that selects the k highest-attention destinations per agent — with a learnable
gate predicting k — achieves task returns within 5 % of dense TarMAC-style
attention while using ≤ N̄ / 4 of the message-channel FLOPs at N ≥ 12.

## Why it's novel
- TarMAC [`das2019tarmac`] uses dense soft attention over all peers; O(N²) cost.
- ATOC [`jiang2018atoc`] uses a learned binary gate per agent; closer to top-1
  with locality, not top-k.
- IC3Net [`singh2018ic3net`] gates communicate-or-not, not who to communicate
  with.
- DGN [`jiang2018dgn`] localises via a fixed neighbour graph; k is not learned.

The novelty is the *adaptive k*: each agent emits a soft over destinations,
takes the top-k (k from a learned head), and back-propagates straight-through
into the gate. No paper in our 64-entry review does this.

## Why it fits Mohammed's profile
Direct extension of MARL-IoTP's learned-attention communication. The
contribution is operationally about sparsifying attention; Mohammed's prior
work motivates the move, and the result speaks to attention-communication
scalability — the natural follow-up.

## Why it fits the target team
Distributed agentic-RL teams care about communication-channel cost: at large
N, dense attention is the rollout bottleneck. A bandwidth-aware attention
variant is operationally relevant for distributed MARL serving (analogous to
how Performers [`choromanski2021performers`] enabled long-context LLM
attention).

## Experimental plan
- **Env:** MPE simple_spread (primary), LBF Foraging (secondary).
- **N sweep:** {3, 6, 12, 25} agents.
- **Baselines:** MAPPO (no comm), MAPPO + dense soft attention comm (our
  TarMAC re-impl), MAPPO + top-k attention with fixed k ∈ {1, 3}, MAPPO +
  adaptive top-k (proposed).
- **Ablations:**
  1. Adaptive k → fixed k=k̄ (the learned mean).
  2. Adaptive k → uniform random k same expectation.
  3. Zero-out the comm output (reduce-to-baseline sanity).
- **Scaling:** primary table over N; secondary over message dimension.
- **Success criterion:** within-5% return parity at N=12 *and* monotone FLOPs
  reduction.

## Compute estimate
Per-run wall-clock at N=12, MPE simple_spread, 1M env steps with 8 parallel
envs ≈ 25 min on one RTX 2080 Ti with our policy size (≤2 GB VRAM). At full
sweep: 4 N's × 5 seeds × 4 conditions = 80 runs × 25 min ≈ 33 GPU-h. Tight in
our 48 GPU-h budget; the adaptive-k ablation pair (5 seeds × 2) eats 4 GPU-h
more — total ~37 GPU-h.

## Risk register
1. **Top-k discrete selection is non-differentiable.** Mitigation: use
   straight-through Gumbel softmax for the destination gate during training;
   evaluate hard top-k at test time and report both.
2. **k̄ collapses to N (no sparsity gained).** Mitigation: add a small budget
   regulariser λ·k̄ that we can tune; report Pareto curve.
3. **Communication is unnecessary on MPE simple_spread.** Mitigation:
   include LBF (where coordination genuinely matters) as secondary.

## Expected venue
NeurIPS 2026 MARL workshop (typical Sep 2026 submission), or AAAI 2027
Distributed AI track. ICLR 2027 if results scale to N=25 cleanly.
