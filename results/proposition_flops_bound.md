# The 2× ceiling on sparse-attention FLOPs savings

A formal statement of the bound reported in D-022, with assumptions made
explicit so that the conditions under which it *fails* are as clear as those
under which it holds. The failure case is not a caveat: it is the design
target for any future method in this line.

Pinned numerically by `tests/test_comm_module.py::test_attention_flop_saving_is_bounded_by_two`
and `::test_structural_sparsity_escapes_the_two_times_bound`.

---

## Setting

One round of scaled dot-product attention over $N$ agents, per batch element.
Query/key/value width $d$. Each agent attends over the $N-1$ peers; self is
masked. Cost is counted in multiply-accumulates (MACs) of the two matrix
products that constitute attention:

$$C_{\text{scores}} = N^2 d, \qquad C_{\text{agg}} = N^2 d,
\qquad C_{\text{dense}} = 2N^2 d.$$

A top-$k$ variant keeping $k$ peers per agent replaces only the second term:

$$C_{\text{top-}k} = \underbrace{N^2 d}_{\text{all scores}} + \underbrace{Nkd}_{\text{aggregate }k}.$$

## Proposition 1 (ceiling)

*For any selection rule that requires the complete score vector of each query,*

$$\frac{C_{\text{dense}}}{C_{\text{top-}k}} \;=\; \frac{2N^2d}{N^2d + Nkd}
\;=\; \frac{2N}{N+k} \;<\; 2 \qquad \text{for all } k \ge 1,$$

*with supremum $2$ approached only as $k/N \to 0$. No choice of $k$, and no
choice of $N$, yields more than a $2\times$ reduction in attention MACs.*

**Proof.** $C_{\text{top-}k} > N^2 d = \tfrac{1}{2} C_{\text{dense}}$ for every
$k \ge 1$, since $Nkd > 0$. The ratio is monotone decreasing in $k$ and tends
to $2$ as $k/N \to 0$. $\square$

**Corollary.** At $k=1$ the saving is still under $2\times$. The familiar
$1 - k/(N-1)$ figure describes the aggregation term alone, which is at most
half of attention. Reporting it as the saving overstates the achievable
reduction by a factor approaching two — before any question of whether the
reduction is realised in time.

## Assumptions, each load-bearing

**A1 — single round.** With $R$ rounds of message passing both terms scale by
$R$ and the ratio is unchanged, so the bound is insensitive to $R$. It does
*not* cover architectures where sparsity changes the number of rounds.

**A2 — data-dependent selection.** The kept set is a function of the scores
$s_i = q_i K^\top$, so all $N$ scores per query must be formed. **This is the
assumption that makes $C_{\text{scores}}$ irreducible, and it is the only one
whose failure lifts the ceiling.**

**A3 — per head.** With $H$ heads of width $d/H$, both terms scale identically;
the bound applies per head and therefore to the sum.

**A4 — projections excluded.** The shared $qkv$ and output projections are
omitted. Including them moves the ratio strictly *closer to 1*, so Proposition
1 remains a valid upper bound on achievable saving.

**A5 — selection cost excluded.** The top-$k$ operation itself
($\Theta(N)$–$\Theta(N\log k)$ comparisons per query) is not counted. Counting
it can only decrease the ratio. Empirically this term dominates: it is
measured at $15.8\times$ the entire dense aggregation at $N{=}512$
(`results/component_decomposition.md`).

## Proposition 2 (escape route)

*If the kept set is determined **structurally** — by information obtainable
without forming the scores, in $o(N^2 d)$ — then the score matrix need not be
computed for discarded pairs, and*

$$C_{\text{struct}} = \underbrace{Nkd}_{\text{scores, kept pairs only}} + \underbrace{Nkd}_{\text{aggregate}} = 2Nkd,
\qquad \frac{C_{\text{dense}}}{C_{\text{struct}}} = \frac{N}{k},$$

*which is unbounded in $N/k$.*

Assumption A2 fails and the ceiling with it. Qualifying selection rules
include a fixed communication topology, a spatial or distance prior, a
locality-sensitive hash or clustering computed in $o(N^2d)$, and a learned
*static* graph. This is precisely why Longformer and BigBird obtain
asymptotic savings where data-dependent top-$k$ cannot: their patterns are
structural, fixed before any score is computed.

**Consequence for this line of work.** A method that selects peers *by
attention score* is bounded at $2\times$ no matter how aggressive its
sparsity. To do better it must decide *where to look before looking* — which
is a different architecture, not a tuning of this one. In an environment with
spatial structure such as MPE, a distance-calibrated selector is the obvious
candidate and is recorded as the primary future-work direction.

## The result that outranks both propositions

Proposition 1 bounds a quantity that does not, in this regime, govern runtime.
Both paths are memory-bound: measured arithmetic intensities sit far below the
machine's ridge point on GPU and CPU alike (`results/roofline.md`), and the
sparse path's intensity is *lower* than the dense path's because top-$k$
selection and the gather move memory for zero arithmetic. Removing MACs from a
memory-bound kernel does not make it faster. That is why the measured
aggregation saving is **negative** — the sparse contraction is slower than the
dense one it replaces — and why no $k$ or $N$ tested produces a crossover.
