# Literature Review — Agentic RL / Multi-Agent RL

**Compiled 2026-05-20.** 64 verified arXiv entries, each with a note file in
`literature/notes/<arxiv-id>.md` and a bibtex entry in `references.bib`. Every
citation in this document maps to a paper whose abstract was fetched from arxiv;
method, benchmark, and limitation lines are written from that abstract plus
the paper's own published self-description. Anything not in `references.bib`
will not appear in the paper.

The four lanes follow the run prompt's structure. The final section turns the
review into a list of concrete tractable contributions, which feeds Phase 3.

---

## Lane A — Classical cooperative MARL and communication learning

**Foundations.** The CTDE (centralised training, decentralised execution) paradigm
crystallised around three closely related ideas: per-agent value decomposition
[`sunehag2017vdn`, `rashid2018qmix`], centralised critics with counterfactual
baselines [`foerster2017coma`, `lowe2017maddpg`], and learned communication
[`foerster2016dial`, `sukhbaatar2016commnet`]. VDN's additive factorisation is the
simplest possible mixer; QMIX adds a non-negative-weight hyper-network on the
global state that preserves a monotonicity guarantee — necessary for tractable
per-agent argmax during decentralised execution. COMA and MADDPG use a centralised
critic and decentralised actors; MADDPG's MPE benchmark suite remains the
default cheapest cooperative testbed.

**Modern cooperative MARL is essentially three families.** (i) Independent learning
with PPO [`dewitt2020ippo`], which de Witt et al. show is remarkably strong on
SMAC despite no centralised information; (ii) MAPPO-family centralised-critic PPO
[`yu2022mappo`], which adds a state-conditioned value head and a handful of
implementation tricks; and (iii) HAPPO/HATRPO [`kuba2021happo`], which extend
trust-region theory to heterogeneous agents via the multi-agent advantage
decomposition lemma. The MAT [`wen2022mat`] paper recasts the joint-action
selection problem as encoder-decoder sequence modelling, achieves linear-in-N
inference, and inherits HAPPO's monotonic improvement guarantee.

**Attention and communication.** TarMAC [`das2019tarmac`] introduced soft attention
over emitted key-value messages for multi-agent communication; IC3Net
[`singh2018ic3net`] added per-agent gating to CommNet's broadcast channel; ATOC
[`jiang2018atoc`] used an attention unit to decide *when* to initiate
communication. DGN [`jiang2018dgn`] views the system as a dynamic graph and uses
graph convolutions. Attention-based value mixers were introduced by Qatten
[`yang2020qatten`], which derives a theoretically motivated multi-head attention
between per-agent utilities and Q_tot. Together, these papers map out the design
space of attention in MARL: attention over messages (TarMAC), attention over
neighbours (DGN), attention over value components (Qatten).

**Representational power of value decomposition.** QTRAN [`son2019qtran`] showed
QMIX/VDN cannot represent value functions whose argmax depends on inter-agent
action interactions; WQMIX [`rashid2020wqmix`] showed that even a weighted
projection that recovers Q* is non-trivial. MAVEN [`mahajan2019maven`]
diagnosed QMIX's exploration failure on hard SMAC maps and proposed latent-conditioned
hierarchical exploration. DOP [`wang2020dop`] integrated value decomposition into
the multi-agent actor-critic for off-policy stability.

**Benchmarks and infrastructure.** SMAC [`samvelyan2019smac`] dominated 2019-2022;
SMACv2 [`ellis2022smacv2`] re-opened the gap by procedural unit generation and
extended partial observability. EPyMARL [`papoudakis2021epymarl`] and MARLlib
[`hu2023marllib`] are the standard reproduction substrates. PettingZoo
[`terry2020pettingzoo`] + SuperSuit [`terry2020supersuit`] standardised the env
API. A 2019 survey of non-stationarity [`papoudakis2019nonstationarity`] gives
the high-level taxonomy.

**Open questions in Lane A.**
- Attention communication remains O(N²) in agents (TarMAC). DGN's locality and
  ATOC's gating each address part of this, but no method scales attention itself.
- Heterogeneous-agent CTDE under HAPPO still updates sequentially — parallel
  monotonic-improvement updates are an open problem.
- Mixers (QMIX, Qatten, WQMIX, QTRAN) are usually evaluated separately from
  communication mechanisms; how they compose is poorly studied.

## Lane B — Agentic RL for LLM agents

**Foundations.** PPO [`schulman2017ppo`] is the universal substrate; InstructGPT
[`ouyang2022instructgpt`] established the 3-stage RLHF pipeline (SFT → RM → PPO).
DPO [`rafailov2023dpo`] eliminates the explicit reward model by exploiting the
closed-form solution to KL-regularised reward maximisation, enabling direct
classification-loss training on preference pairs. GRPO [`shao2024grpo`] drops the
critic from PPO in favour of group-relative advantages; cheaper memory, suitable
for verifiable-reward domains. Process reward models [`lightman2023letsverify`]
provide step-level credit assignment via 800k human-labelled MATH traces.

**Inference-time agentic patterns.** ReAct [`yao2022react`] interleaves chain-of-
thought reasoning with environment actions in a few-shot prompted loop;
Reflexion [`shinn2023reflexion`] replaces gradient updates with linguistic
self-critique across trials; Voyager [`wang2023voyager`] builds a skill library
in Minecraft with curriculum and self-verification. Toolformer
[`schick2023toolformer`] is the first self-supervised approach to tool API use.

**Multi-agent LLM systems.** AutoGen [`wu2023autogen`] and MetaGPT
[`hong2023metagpt`] define multi-agent LLM application frameworks based on
conversation and SOP-encoded role assignment respectively. CAMEL
[`li2023camel`] generates instruction data through inception-prompted role-play.
Multi-agent debate [`du2023multiagentdebate`] improves factuality and reasoning
through repeated critique rounds. None of these train agents end-to-end with RL;
they treat the LLM as a fixed policy and design coordination protocols on top.

**Self-play and bootstrapped data.** SPIN [`chen2024spin`] trains an LLM
iteratively to distinguish its own samples from human-annotated data, recovering
DPO-quality performance without external preferences. AgentTuning
[`zeng2023agenttuning`] mixes agent trajectories with general SFT data to
preserve broad LM capability while gaining agentic skills.

**Evaluation.** AgentBench [`liu2023agentbench`] is the multi-environment LLM-as-
agent suite; WebArena [`zhou2023webarena`] provides a realistic web environment
(GPT-4 ≈ 14% success vs human 78%); ALFWorld [`shridhar2020alfworld`] aligns
text-based and embodied household tasks; BabyAI [`chevalierboisvert2018babyai`]
is the lightest-weight instruction-following grid-world.

**Open questions in Lane B.**
- Process credit assignment for **multi-step tool agents** is currently solved by
  PRMs that require human annotation [`lightman2023letsverify`]. Self-supervised
  process supervision is open.
- **Multi-agent LLM training**: AutoGen/MetaGPT use fixed LLMs; no open-source
  pipeline jointly RL-fines-tunes multiple cooperating LLM roles.
- **Sample-efficient verbal RL** vs gradient RL trade-offs at scale are
  empirically unclear.

## Lane C — Distributed RL training infrastructure

**Classical distributed RL.** IMPALA [`espeholt2018impala`] introduced the
decoupled actor-learner architecture with V-trace off-policy correction; Ape-X
[`horgan2018apex`] separated CPU actors from a GPU learner via prioritised
replay; SEED RL [`espeholt2020seedrl`] moved inference to a centralised
accelerator server. RLlib [`liang2017rllib`] (Ray-based) and Acme
[`hoffman2020acme`] (Launchpad/Reverb-based) are the dominant frameworks.
CleanRL [`huang2022cleanrl`] provides single-file reference implementations.

**LLM-RL serving stack.** vLLM with PagedAttention [`kwon2023vllm`] is the
inference engine that the modern RLHF stack builds on, providing 2-4× throughput
via KV-cache paging and shared prefixes. HybridFlow [`sheng2024hybridflow`]
combines single- and multi-controller paradigms for the RLHF dataflow, with a
3D-HybridEngine that reshards the actor between rollout and training phases —
1.53-20.57× throughput improvement. OpenRLHF [`hu2024openrlhf`] is the
open-source Ray + vLLM + DeepSpeed implementation; DeepSpeed-Chat
[`yao2023deepspeedchat`] is Microsoft's industrial reference.

**Attention scaling.** Performers [`choromanski2021performers`] approximate the
softmax kernel with random features in linear time and space. This is one
candidate mechanism if we want to scale agent-attention beyond O(N²).

**Open questions in Lane C.**
- Distributed **MARL** (as opposed to single-agent or LLM-RL) infra is
  conspicuously absent in the recent literature — most papers reuse Ape-X- or
  IMPALA-style designs with shared parameters.
- The **rollout/training resharding** strategies from HybridFlow have no analogue
  on the MARL side, where per-agent critic placement and actor placement are
  similarly composable.
- Centralised inference (SEED-RL) and centralised critics (QMIX/MAPPO) overlap
  conceptually but are not unified in a single framework.

## Lane D — Game-theoretic and large-scale MARL

**Foundations.** Mean-field MARL [`yang2018meanfield`] approximates many-agent
dynamics by the mean effect of neighbours/population, giving tractable
fixed-point updates at the cost of homogeneity assumptions. PBT
[`jaderberg2017pbt`] is the population-based training pattern (exploit/explore)
that AlphaStar and OpenAI Five [`berner2019openai5`] use at scale.

**Emergent behaviour and benchmarks.** Hide-and-seek [`baker2020hideandseek`]
showed six emergent strategy phases via self-play autocurricula; Hanabi
[`bard2019hanabi`] is a clean cooperative theory-of-mind benchmark; Melting Pot
[`leibo2021meltingpot`] is the largest mixed-motive evaluation suite. The
Overcooked-AI work [`carroll2019overcooked`] is the canonical small-scale
human-AI coordination benchmark.

**Open questions in Lane D.**
- Mean-field approximations assume neighbour-symmetry. Heterogeneous large-N
  cooperation is mostly unsolved.
- Many-agent attention without O(N²) is the practical scalability bottleneck.
- Evaluation: most cooperative MARL benchmarks plateau at 5-10 agents; the field
  lacks a low-cost benchmark for N >> 10.

---

## Open problems and tractable contributions

For each open problem we list (1) the gap in one sentence, (2) the evidence that
it is open (which paper said so), (3) the dominant approach today, and (4) a
compute estimate. The Phase 3 proposals will pick from this menu.

### OP-1. Subquadratic agent-attention for many-agent communication
- **Gap.** TarMAC-style attention over messages is O(N²) in agents; ATOC's
  neighbour gating and DGN's graph locality are application-specific.
- **Evidence open.** TarMAC limitations section [`das2019tarmac`]; no
  Performer-like attention applied to MARL communication exists in the surveyed
  literature.
- **Dominant approach.** Quadratic attention + sparse gating heuristics.
- **Compute estimate.** MPE simple_spread at N ∈ {3, 6, 12, 25}, MAPPO + a
  comm module: ~0.5 GPU-h per (N, seed) — 5 seeds × 4 N's × 2 conditions = 20
  GPU-h. Tight fit for our budget.

### OP-2. Composable attention-mixer + attention-comm
- **Gap.** Qatten studies attention-as-mixer, TarMAC studies attention-as-
  channel, but no paper studies both together — they could either be redundant
  or complementary.
- **Evidence open.** Neither paper compares against the other; the modern MAT
  [`wen2022mat`] subsumes the mixer but not learned-comm.
- **Dominant approach.** Pick one; don't compose.
- **Compute estimate.** MPE + LBF, MAPPO baseline + the 2×2 cross of
  {attention-mixer ∈ {on, off}} × {attention-comm ∈ {on, off}}, 5 seeds — about
  15 GPU-h.

### OP-3. Heterogeneous attention communication
- **Gap.** Communication papers assume parameter sharing across agents; HAPPO
  shows heterogeneity is feasible but doesn't include comm.
- **Evidence open.** [`kuba2021happo`] limitations + [`das2019tarmac`] design.
- **Dominant approach.** Parameter-shared comm head.
- **Compute estimate.** Multi-Agent MuJoCo or LBF with heterogeneous roles, 5
  seeds × 2 conditions ≈ 15 GPU-h.

### OP-4. Reduce-to-baseline ablation of communication
- **Gap.** Comm-MARL papers rarely run the simplest ablation: zero-out the comm
  output and check that the agent reduces to the baseline. Without it, gains may
  come from architecture, not communication content.
- **Evidence open.** Inspection of [`das2019tarmac`], [`jiang2018atoc`], and
  [`singh2018ic3net`] — none reports this ablation cleanly.
- **Dominant approach.** No standard protocol.
- **Compute estimate.** Adds 5 seeds × 1 condition to any of the above plans;
  cheap (~3 GPU-h).

### OP-5. Communication credit assignment via process rewards
- **Gap.** Comm-MARL methods learn what to send via end-to-end task reward.
  Process-reward methods [`lightman2023letsverify`] give per-step credit. No
  paper applies process rewards to MARL communication.
- **Evidence open.** [`lightman2023letsverify`] is LLM-only; MARL comm papers
  use task reward only.
- **Dominant approach.** End-to-end task reward through messages.
- **Compute estimate.** Requires a verifier signal — non-trivial to construct
  for MPE; better in agentic-LLM settings. ~30 GPU-h if pursued.

### OP-6. Mean-field-meets-attention for many-agent cooperation
- **Gap.** Mean-field MARL [`yang2018meanfield`] uses raw mean neighbour action;
  attention could replace the mean with a learned weighted aggregator.
- **Evidence open.** Mean-field assumption is the named limitation in
  [`yang2018meanfield`].
- **Dominant approach.** Plain mean aggregation.
- **Compute estimate.** Battle game / Ising — heavier; ~30-40 GPU-h.

### OP-7. Sparse communication via top-k attention
- **Gap.** Soft attention over all agents is expensive at large N. Top-k
  attention could cap per-agent message budget, with theoretical guarantees on
  the gap to soft attention.
- **Evidence open.** ATOC's gating is binary, not top-k; no top-k attention in
  surveyed comm-MARL.
- **Dominant approach.** Soft attention.
- **Compute estimate.** MPE scaling sweep at N ∈ {6, 12, 25}, 5 seeds × 3
  conditions ≈ 12-15 GPU-h.

### OP-8. Compositional generalisation of communication protocols
- **Gap.** Most comm methods train and test at the same N. Whether learned
  protocols generalise to unseen agent counts is rarely tested. MAT
  [`wen2022mat`] reports few-shot transfer for the policy but not for the comm
  channel.
- **Evidence open.** MAT generalisation section + absence in TarMAC/ATOC.
- **Dominant approach.** Retrain at the new N.
- **Compute estimate.** ~10 GPU-h for a generalisation table.

### OP-9. Distributed rollout for many-agent cooperative MARL
- **Gap.** SEED RL [`espeholt2020seedrl`] / IMPALA address single-agent
  distributed rollout. For MARL, the standard is "spin up M envs in parallel,
  one process each." How to do centralised-inference-style consolidation for
  per-agent policies at large N is an open infra question.
- **Evidence open.** No paper in our survey addresses this.
- **Dominant approach.** Per-env per-agent inference; serial.
- **Compute estimate.** Mostly engineering; a 10× wall-time improvement at the
  same statistical FLOPS would be a strong infra contribution but does not
  trivially fit our hardware budget.

### OP-10. Communication budget as an explicit RL objective
- **Gap.** Comm methods optimise task return; communication itself has a cost
  (bandwidth, latency) that is rarely in the objective. A constrained-MDP
  formulation with a budget on bits-per-step is open.
- **Evidence open.** None of the surveyed papers report a bits-per-step budget.
- **Dominant approach.** No budget.
- **Compute estimate.** Cheap (uses any MARL setup) — ~8 GPU-h.

### OP-11. RL-fine-tuning of multi-agent LLM systems
- **Gap.** AutoGen [`wu2023autogen`], MetaGPT [`hong2023metagpt`], Multi-Agent
  Debate [`du2023multiagentdebate`] treat LLMs as fixed policies. RL-fine-tuning
  multiple cooperating LLM roles is open.
- **Evidence open.** Survey of those papers' abstracts; no joint RL training.
- **Dominant approach.** Prompt engineering.
- **Compute estimate.** Requires multiple LLMs and verifiable rewards;
  prohibitive on our budget — 200+ GPU-h for any meaningful experiment.

### OP-12. Heterogeneous-policy MARL with shared communication channel
- **Gap.** Combine HAPPO's heterogeneous-agent guarantee with TarMAC's learned
  comm such that the channel itself is shared (parameter-shared) while policies
  are heterogeneous.
- **Evidence open.** Composition not reported.
- **Dominant approach.** Either fully shared or fully separate.
- **Compute estimate.** Multi-Agent MuJoCo at 3-6 agents, 5 seeds × 2 conditions
  ≈ 10-15 GPU-h.

---

## Summary table

| OP   | Gap                                                | Compute (GPU-h) | Fit to MARL-IoTP / target team |
|------|----------------------------------------------------|-----------------|-------------------------------|
| OP-1 | Subquadratic agent-attention                       | 20              | High / High                   |
| OP-2 | Compose attention-mixer + attention-comm           | 15              | High / Medium                 |
| OP-3 | Heterogeneous attention communication              | 15              | High / Medium                 |
| OP-4 | Standard reduce-to-baseline ablation               | 3               | Low / Low                     |
| OP-5 | Process rewards for comm                           | 30              | Medium / Medium               |
| OP-6 | Mean-field × attention                             | 35              | Medium / High                 |
| OP-7 | Top-k attention for sparse comm                    | 15              | High / High                   |
| OP-8 | Compositional comm generalisation across N         | 10              | High / Medium                 |
| OP-9 | Distributed rollout for many-agent MARL            | ≥48 (infra)     | Medium / Very High            |
| OP-10| Comm-budget constrained MDP                        | 8               | Medium / Medium               |
| OP-11| RL-fine-tune multi-agent LLM systems               | ≥200            | Medium / Very High            |
| OP-12| Heterogeneous policy + shared comm                 | 15              | High / Medium                 |

The high-fit, low-compute candidates for Phase 3 are **OP-1, OP-2, OP-7, OP-8**.
Each can be done in our budget, each extends Mohammed's MARL-IoTP attention
communication direction, and each tells a clean story to the target team.
