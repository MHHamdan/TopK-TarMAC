"""Generate literature/references.bib + literature/notes/<id>.md from a single
structured data source.

Every entry below was created from a verified arxiv abstract fetched on 2026-05-20.
Method, benchmarks, and limitations summaries are written from the paper's own text;
no claim appears in the note that is not in the paper.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIT = REPO / "literature"
NOTES = LIT / "notes"
NOTES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data: 49 verified papers
# Each entry: arxiv_id, key, lane (A/B/C/D), title, authors, year, abstract,
# method (5 lines max), benchmarks_results, limitations, connection.
# `connection` is one line — empty string allowed.
# ---------------------------------------------------------------------------

PAPERS = [
    # --------------------------------- Lane A ---------------------------------
    dict(
        id="2103.01955", key="yu2022mappo", lane="A",
        title="The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games",
        authors=("Chao Yu and Akash Velu and Eugene Vinitsky and Jiaxuan Gao and "
                 "Yu Wang and Alexandre Bayen and Yi Wu"),
        year=2022, venue="NeurIPS Datasets and Benchmarks",
        abstract=(
            "PPO-based multi-agent algorithms achieve surprisingly strong performance "
            "in MPE, SMAC, GRF, and Hanabi with minimal hyperparameter tuning and "
            "without domain-specific architectures, matching or beating off-policy "
            "methods in final return and sample efficiency."
        ),
        method=(
            "MAPPO = independent PPO actors + a centralised value function over the "
            "global state, shared parameters across homogeneous agents, GAE, and a "
            "small set of implementation tricks (value clipping, advantage "
            "normalisation, large batch + multiple PPO epochs)."
        ),
        benchmarks=(
            "MPE, SMAC, Google Research Football, Hanabi. Achieves SOTA-competitive "
            "win-rates on 14/23 SMAC maps studied; matches RODE/QMIX on most maps."
        ),
        limitations=(
            "Strong only on cooperative homogeneous settings; lots of tuning around "
            "tricks; no heterogeneous-agent guarantees."
        ),
        connection=(
            "Default cooperative-MARL baseline — direct candidate for our reproduction "
            "baseline."
        ),
    ),
    dict(
        id="1803.11485", key="rashid2018qmix", lane="A",
        title="QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL",
        authors=("Tabish Rashid and Mikayel Samvelyan and Christian Schroeder de Witt "
                 "and Gregory Farquhar and Jakob Foerster and Shimon Whiteson"),
        year=2018, venue="ICML",
        abstract=(
            "QMIX is a value-based CTDE method that mixes per-agent utilities into a "
            "joint Q via a non-linear monotonic mixer conditioned on the global state, "
            "enabling tractable joint-action argmax and consistency between "
            "centralised and decentralised policies."
        ),
        method=(
            "Per-agent Q networks output utilities Q_i(τ_i,a_i); a hyper-network "
            "produces non-negative weights of a feed-forward mixer that combines "
            "utilities into Q_tot; monotonicity guarantees argmax_a Q_tot = "
            "(argmax_{a_i} Q_i)_i so decentralised greedy actions remain optimal."
        ),
        benchmarks="SMAC StarCraft II micromanagement; significantly outperforms VDN, IQL.",
        limitations=(
            "Monotonicity restricts representable joint value functions; cannot model "
            "non-monotonic coordination (see WQMIX, QTRAN)."
        ),
        connection=(
            "Canonical value-decomposition CTDE baseline; also a target of the MARL-IoTP "
            "attention-comm hypothesis."
        ),
    ),
    dict(
        id="1705.08926", key="foerster2017coma", lane="A",
        title="Counterfactual Multi-Agent Policy Gradients",
        authors=("Jakob Foerster and Gregory Farquhar and Triantafyllos Afouras "
                 "and Nantas Nardelli and Shimon Whiteson"),
        year=2017, venue="AAAI",
        abstract=(
            "COMA uses a centralised critic and decentralised actors; the critic "
            "computes a counterfactual baseline that marginalises out a single agent's "
            "action to attribute credit, addressing the multi-agent credit-assignment "
            "problem."
        ),
        method=(
            "Single critic Q(s,a) shared across agents; per-agent advantage A_i = "
            "Q(s,a) - Σ_{a_i'} π_i(a_i'|τ_i) Q(s,(a_i', a_{-i})); efficient because the "
            "critic outputs |A_i| values in one forward pass."
        ),
        benchmarks="StarCraft I micromanagement (decentralised, partial observability).",
        limitations="On-policy, fragile in sparse-reward; eclipsed by MAPPO/QMIX numerically.",
        connection=(
            "Origin of multi-agent credit assignment via counterfactual baselines — "
            "conceptually related to attention-weighted message routing."
        ),
    ),
    dict(
        id="1706.02275", key="lowe2017maddpg", lane="A",
        title="Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
        authors=("Ryan Lowe and Yi Wu and Aviv Tamar and Jean Harb and Pieter Abbeel "
                 "and Igor Mordatch"),
        year=2017, venue="NeurIPS",
        abstract=(
            "MADDPG extends DDPG to multi-agent by giving each agent's critic access to "
            "the joint observation and joint action; an ensemble of policies improves "
            "robustness against non-stationary co-players."
        ),
        method=(
            "Per-agent actor μ_i(o_i); per-agent centralised critic Q_i(x, a_1..a_n); "
            "off-policy DDPG-style updates with replay; ensemble policies sampled at "
            "training time to combat overfit to specific co-players."
        ),
        benchmarks=(
            "Multi-Particle Environments (MPE) — simple_spread, simple_speaker_listener, "
            "simple_push, predator-prey-style tasks. Beats independent DDPG."
        ),
        limitations=(
            "Doesn't scale beyond ~10 agents; sensitive to hyperparameters; "
            "deterministic policy gradients problematic in fully cooperative settings."
        ),
        connection=(
            "Defined the MPE benchmark suite — our likely first-stage testbed."
        ),
    ),
    dict(
        id="1706.05296", key="sunehag2017vdn", lane="A",
        title="Value-Decomposition Networks for Cooperative Multi-Agent Learning",
        authors=("Peter Sunehag and Guy Lever and Audrunas Gruslys and Wojciech Marian "
                 "Czarnecki and Vinicius Zambaldi and Max Jaderberg and Marc Lanctot and "
                 "Nicolas Sonnerat and Joel Z. Leibo and Karl Tuyls and Thore Graepel"),
        year=2017, venue="AAMAS",
        abstract=(
            "VDN factorises the team Q-function as a sum of per-agent utilities; "
            "per-agent argmax then maximises Q_tot. Mitigates the lazy-agent problem of "
            "centralised Q-learning under partial observability."
        ),
        method=(
            "Q_tot(τ,a) = Σ_i Q_i(τ_i,a_i); per-agent networks trained by backprop "
            "through the sum; weight sharing + role information improve generalisation."
        ),
        benchmarks="2D partially-observable predator-prey, switch riddle, key-and-door.",
        limitations="Additivity is too restrictive for many cooperative tasks (QMIX, WQMIX).",
        connection="Sets up the linear baseline for value-decomposition that QMIX/WQMIX/QTRAN refine.",
    ),
    dict(
        id="1810.11187", key="das2019tarmac", lane="A",
        title="TarMAC: Targeted Multi-Agent Communication",
        authors=("Abhishek Das and Théophile Gervet and Joshua Romoff and Dhruv Batra "
                 "and Devi Parikh and Michael Rabbat and Joelle Pineau"),
        year=2019, venue="ICML",
        abstract=(
            "Agents learn what to communicate and to whom via a soft attention "
            "mechanism over outgoing key/value pairs; multi-round communication "
            "improves coordination in partially observable cooperative tasks."
        ),
        method=(
            "Each agent emits a key/value pair; receivers form queries; messages are "
            "aggregated by scaled dot-product attention. Multiple rounds before action; "
            "trained end-to-end with task reward only."
        ),
        benchmarks=(
            "SHAPES, MPE traffic-junction, House3D, Predator-prey; consistent gains "
            "from targeted vs broadcast comm, especially at high agent count."
        ),
        limitations=(
            "Quadratic O(N²) communication cost; assumes a fully-connected "
            "communication graph; small numbers of agents in evaluation."
        ),
        connection=(
            "Direct ancestor of MARL-IoTP's attention communication; quadratic cost is "
            "the gap our scaling work would address."
        ),
    ),
    dict(
        id="1605.06676", key="foerster2016dial", lane="A",
        title="Learning to Communicate with Deep Multi-Agent Reinforcement Learning",
        authors="Jakob N. Foerster and Yannis M. Assael and Nando de Freitas and Shimon Whiteson",
        year=2016, venue="NeurIPS",
        abstract=(
            "Two methods for end-to-end learned communication: RIAL (reinforced "
            "inter-agent learning) treats messages as discrete actions; DIAL "
            "(differentiable) backpropagates through a noisy continuous channel during "
            "training and discretises at execution."
        ),
        method=(
            "Shared deep Q-network outputs both environment-action and message; DIAL "
            "passes continuous messages with noise during training (gradients flow), "
            "switches to discrete messages at test time. CTDE with parameter sharing."
        ),
        benchmarks="Switch riddle, MNIST games — small puzzles testing protocol emergence.",
        limitations="Toy benchmarks; small N; discrete-channel discretisation gap.",
        connection="Foundational paper that motivates every modern attention-comm method.",
    ),
    dict(
        id="2011.09533", key="dewitt2020ippo", lane="A",
        title="Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?",
        authors=("Christian Schroeder de Witt and Tarun Gupta and Denys Makoviichuk and "
                 "Viktor Makoviychuk and Philip H. S. Torr and Mingfei Sun and "
                 "Shimon Whiteson"),
        year=2020, venue="arXiv preprint",
        abstract=(
            "Independent PPO (IPPO), where each agent estimates its own local value "
            "function with no centralised critic, matches or beats centralised joint-"
            "value-function methods on SMAC with little tuning. Suggests robustness to "
            "non-stationarity matters more than centralised information at SMAC's scale."
        ),
        method=(
            "PPO per agent with shared parameters and only local observations + actions "
            "— no centralised critic, no value decomposition."
        ),
        benchmarks="SMAC v1; matches or beats QMIX, MAVEN, QTRAN, COMA on most maps.",
        limitations=(
            "Empirical; doesn't theoretically explain why IPPO works; sensitive to PPO "
            "implementation details (clipping, advantage normalisation)."
        ),
        connection="Pure-independence baseline against which any communication method must show value.",
    ),
    dict(
        id="2109.11251", key="kuba2021happo", lane="A",
        title="Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning",
        authors=("Jakub Grudzien Kuba and Ruiqing Chen and Muning Wen and Ying Wen and "
                 "Fanglei Sun and Jun Wang and Yaodong Yang"),
        year=2021, venue="ICLR 2022",
        abstract=(
            "Extends trust-region theory to multi-agent: a multi-agent advantage "
            "decomposition lemma + sequential update scheme yield HATRPO and HAPPO, "
            "the first MARL algorithms with monotonic-improvement guarantees that work "
            "with heterogeneous, non-parameter-shared agents."
        ),
        method=(
            "Agents update sequentially in a random order each iteration; each agent "
            "performs a trust-region step on the advantage decomposed conditional on "
            "preceding agents' new policies. No parameter sharing required."
        ),
        benchmarks="Multi-Agent MuJoCo, SMAC; outperforms MAPPO, MADDPG, QMIX, IPPO.",
        limitations="Sequential update scheme limits parallelism; still O(N) trust-region computations.",
        connection="Theoretical anchor for any heterogeneous-agent communication method we propose.",
    ),
    dict(
        id="2205.14953", key="wen2022mat", lane="A",
        title="Multi-Agent Reinforcement Learning is a Sequence Modeling Problem",
        authors=("Muning Wen and Jakub Grudzien Kuba and Runji Lin and Weinan Zhang and "
                 "Ying Wen and Jun Wang and Yaodong Yang"),
        year=2022, venue="NeurIPS",
        abstract=(
            "Multi-Agent Transformer (MAT) treats joint-action selection as encoder-"
            "decoder sequence-to-sequence translation: encode agent observations, "
            "autoregressively decode actions in agent order. Linear complexity in N; "
            "inherits the multi-agent advantage decomposition guarantee."
        ),
        method=(
            "Encoder: standard self-attention over agent obs → token embeddings + "
            "value head. Decoder: autoregressive action generation conditioned on the "
            "encoded observations and previously decoded actions. Trained on-policy "
            "with PPO-style updates."
        ),
        benchmarks=(
            "SMAC, Multi-Agent MuJoCo, Bi-Dex Hands, GRF. Beats MAPPO/HAPPO; "
            "few-shot transfers across agent counts."
        ),
        limitations=(
            "Sequential decoding is unfriendly to inference parallelism; transformer "
            "is heavyweight for small N."
        ),
        connection=(
            "Closest precursor to attention-based scalable cooperation — direct target "
            "for our scaling/comm contribution."
        ),
    ),
    dict(
        id="1902.04043", key="samvelyan2019smac", lane="A",
        title="The StarCraft Multi-Agent Challenge",
        authors=("Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt "
                 "and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and "
                 "Chia-Man Hung and Philip H. S. Torr and Jakob Foerster and "
                 "Shimon Whiteson"),
        year=2019, venue="AAMAS",
        abstract=(
            "Introduces SMAC, a cooperative MARL benchmark based on StarCraft II "
            "micromanagement, with decentralised partial observability per unit. "
            "Provides PyMARL, a reference framework with QMIX/COMA/VDN baselines."
        ),
        method="Benchmark + framework — not an algorithm.",
        benchmarks="14 maps spanning easy/hard/super-hard cooperative micromanagement.",
        limitations="Recent work (SMACv2) shows open-loop policies do non-trivially well.",
        connection="Optional stretch benchmark; out of compute budget for this run.",
    ),
    dict(
        id="2212.07489", key="ellis2022smacv2", lane="A",
        title="SMACv2: An Improved Benchmark for Cooperative Multi-Agent RL",
        authors=("Benjamin Ellis and Jonathan Cook and Skander Moalla and Mikayel "
                 "Samvelyan and Mingfei Sun and Anuj Mahajan and Jakob N. Foerster and "
                 "Shimon Whiteson"),
        year=2022, venue="NeurIPS Datasets and Benchmarks",
        abstract=(
            "SMACv2 procedurally generates scenarios and adds an extended partial-"
            "observability challenge, forcing closed-loop policies. Re-establishes a "
            "meaningful gap between SOTA and ceiling."
        ),
        method="Benchmark — procedural unit composition + EPO observation mask.",
        benchmarks="Reports MAPPO/QMIX baselines; gap to ceiling re-opened.",
        limitations="StarCraft II install still required; same SC2 dependency as SMAC.",
        connection="Stretch benchmark if compute permits; otherwise a positioning reference.",
    ),
    dict(
        id="1703.10069", key="peng2017bicnet", lane="A",
        title="Multiagent Bidirectionally-Coordinated Nets: Emergence of Human-level Coordination in Learning to Play StarCraft Combat Games",
        authors=("Peng Peng and Ying Wen and Yaodong Yang and Quan Yuan and Zhenkun Tang "
                 "and Haitao Long and Jun Wang"),
        year=2017, venue="arXiv preprint",
        abstract=(
            "BiCNet maintains a bidirectional RNN across agents for coordinated action "
            "selection in a vectorised actor-critic. Demonstrates emergent multi-unit "
            "coordination on StarCraft I combat scenarios."
        ),
        method=(
            "Bi-LSTM links all agents; each cell outputs that agent's action; "
            "actor-critic with central state value. Effectively a sequence model over "
            "agents."
        ),
        benchmarks="StarCraft I unit combat scenarios; arbitrary team sizes.",
        limitations="Recurrent across agents → O(N) sequential decode; older benchmark.",
        connection="Precursor to MAT's sequence-modelling view of MARL.",
    ),
    dict(
        id="1812.09755", key="singh2018ic3net", lane="A",
        title="Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks",
        authors="Amanpreet Singh and Tushar Jain and Sainbayar Sukhbaatar",
        year=2018, venue="ICLR",
        abstract=(
            "IC3Net adds a per-agent gating mechanism to CommNet so agents can choose "
            "whether to communicate at all; works in cooperative, mixed, and "
            "competitive settings; per-agent rewards improve credit assignment."
        ),
        method=(
            "CommNet backbone (mean message across agents) + a binary gate per agent "
            "controlling whether the agent emits a message; per-agent rewards in the "
            "loss."
        ),
        benchmarks="Traffic-junction, predator-prey, StarCraft BroodWars combat.",
        limitations=(
            "Gating is binary; broadcast topology limits targeted exchanges; older "
            "policy backbone."
        ),
        connection="Gating is conceptually a sparse-attention prior we may reuse.",
    ),
    dict(
        id="1805.07733", key="jiang2018atoc", lane="A",
        title="Learning Attentional Communication for Multi-Agent Cooperation",
        authors="Jiechuan Jiang and Zongqing Lu",
        year=2018, venue="NeurIPS",
        abstract=(
            "ATOC learns an attention unit that decides when communication is needed "
            "and an LSTM-based communication channel that aggregates messages from "
            "selected nearby agents — improving large-scale cooperation."
        ),
        method=(
            "Per-agent attention unit decides whether to initiate communication; a "
            "bidirectional LSTM communicates among initiator's neighbours; outputs "
            "thought vectors that condition the policy. DDPG actor-critic."
        ),
        benchmarks="Cooperative navigation, predator-prey, jungle (custom).",
        limitations="Locality assumption (neighbour set); LSTM channel is sequential.",
        connection="Direct precedent for adaptive sparse communication.",
    ),
    dict(
        id="1605.07736", key="sukhbaatar2016commnet", lane="A",
        title="Learning Multiagent Communication with Backpropagation",
        authors="Sainbayar Sukhbaatar and Arthur Szlam and Rob Fergus",
        year=2016, venue="NeurIPS",
        abstract=(
            "CommNet introduces a continuous, backprop-trainable communication "
            "channel between agents, learned end-to-end alongside the policy. Yields "
            "improved performance in fully cooperative tasks and sometimes "
            "interpretable strategies."
        ),
        method=(
            "Each agent emits a continuous message vector; messages are averaged and "
            "passed back to all agents; multiple rounds of message-passing inside one "
            "decision."
        ),
        benchmarks="Traffic-junction, lever-pulling, combat.",
        limitations=(
            "Mean-aggregation loses identity; not addressable; broadcast bandwidth "
            "scales linearly."
        ),
        connection="The mean-aggregation baseline our attention method should beat.",
    ),
    dict(
        id="1810.09202", key="jiang2018dgn", lane="A",
        title="Graph Convolutional Reinforcement Learning",
        authors="Jiechuan Jiang and Chen Dun and Tiejun Huang and Zongqing Lu",
        year=2018, venue="ICLR 2020",
        abstract=(
            "DGN treats the multi-agent system as a dynamic graph and applies graph "
            "convolutions with relation kernels to learn coordination. Temporal "
            "relation regularisation stabilises cooperation."
        ),
        method=(
            "K-layer GCN over the agent neighbourhood graph; attention-like relation "
            "kernels weight neighbours; Q-learning with target nets."
        ),
        benchmarks="Cooperative navigation, jungle, battle (large-scale).",
        limitations="Requires defining a graph (often spatial proximity); discrete actions.",
        connection="Graph-attention prior we can extend with learned communication.",
    ),
    dict(
        id="1910.07483", key="mahajan2019maven", lane="A",
        title="MAVEN: Multi-Agent Variational Exploration",
        authors=("Anuj Mahajan and Tabish Rashid and Mikayel Samvelyan and "
                 "Shimon Whiteson"),
        year=2019, venue="NeurIPS",
        abstract=(
            "MAVEN hybridises value- and policy-based MARL via a shared latent variable "
            "controlling per-episode joint exploration. Addresses QMIX's restricted "
            "expressiveness and poor exploration."
        ),
        method=(
            "Hierarchical policy samples a latent z each episode; QMIX is trained "
            "conditional on z; an inverse model + mutual-information loss encourages "
            "z to control distinguishable behaviour."
        ),
        benchmarks="SMAC hard maps where QMIX stalls.",
        limitations=(
            "Adds latent-policy complexity; benefits concentrated on hard-exploration "
            "maps."
        ),
        connection="Identifies QMIX's representational limit — relevant to value-mixer choice.",
    ),
    dict(
        id="2006.10800", key="rashid2020wqmix", lane="A",
        title="Weighted QMIX: Expanding Monotonic Value Function Factorisation",
        authors="Tabish Rashid and Gregory Farquhar and Bei Peng and Shimon Whiteson",
        year=2020, venue="NeurIPS",
        abstract=(
            "QMIX's projection step minimises an unweighted error over joint actions, "
            "which can fail to recover the optimal policy even given Q*. WQMIX places "
            "extra weight on better joint actions; OW-QMIX and CW-QMIX provably "
            "recover the optimal greedy action."
        ),
        method=(
            "Weighted projection: w(s,a) = α + (1-α)·1[a=a*]; α ∈ {fixed, learned}. "
            "Two practical variants: optimistically- and centrally-weighted."
        ),
        benchmarks="Predator-prey, SMAC hard maps; outperforms QMIX, MAVEN.",
        limitations=(
            "Weighting heuristic; effectiveness depends on accurate identification of "
            "good joint actions."
        ),
        connection="Useful drop-in replacement if our method composes with QMIX-family mixers.",
    ),
    dict(
        id="1905.05408", key="son2019qtran", lane="A",
        title="QTRAN: Learning to Factorize with Transformation for Cooperative MARL",
        authors=("Kyunghwan Son and Daewoo Kim and Wan Ju Kang and David Earl Hostallero "
                 "and Yung Yi"),
        year=2019, venue="ICML",
        abstract=(
            "QTRAN drops VDN/QMIX's additivity/monotonicity constraints and instead "
            "transforms the joint Q into a factorisable surrogate with identical "
            "optimal actions. Provably represents a strictly larger class of MARL "
            "value functions."
        ),
        method=(
            "Joint Q_jt + per-agent utilities Q_i; auxiliary loss enforces "
            "Σ_i Q_i(τ,a_i) ≥ Q_jt(τ,a) with equality at the joint argmax."
        ),
        benchmarks="Multi-step matrix games where QMIX fails; SMAC.",
        limitations="Auxiliary losses fragile; QPLEX/WQMIX often outperform empirically.",
        connection="Anchor for the representational-power discussion in related work.",
    ),
    dict(
        id="2002.03939", key="yang2020qatten", lane="A",
        title="Qatten: A General Framework for Cooperative Multiagent RL",
        authors=("Yaodong Yang and Jianye Hao and Ben Liao and Kun Shao and Guangyong "
                 "Chen and Wulong Liu and Hongyao Tang"),
        year=2020, venue="arXiv preprint",
        abstract=(
            "Derives a general expression of Q_tot in terms of per-agent Q_i and "
            "approximates the coefficients with multi-head attention over agent "
            "embeddings, yielding an attention-based value decomposition."
        ),
        method=(
            "Multi-head attention over agent observations produces per-head weights "
            "λ_i^h that combine Q_i into Q_tot. Theoretical Taylor-expansion grounding."
        ),
        benchmarks="SMAC — outperforms QMIX, VDN, QTRAN on most maps.",
        limitations="Still monotonic-like under non-negative weights; assumes shared state access.",
        connection="Direct attention-based mixer we may compose with or compare to.",
    ),
    dict(
        id="2007.12322", key="wang2020dop", lane="A",
        title="Off-Policy Multi-Agent Decomposed Policy Gradients",
        authors=("Yihan Wang and Beining Han and Tonghan Wang and Heng Dong and "
                 "Chongjie Zhang"),
        year=2020, venue="ICLR 2021",
        abstract=(
            "DOP integrates value-decomposition into the multi-agent actor-critic, "
            "supporting off-policy learning and curing the centralised-decentralised "
            "mismatch and credit-assignment issues of MAPG methods."
        ),
        method=(
            "Decomposed critic Q_tot = Σ_i k_i Q_i with k_i non-negative; per-agent "
            "actors trained on local advantages; off-policy with target nets and replay."
        ),
        benchmarks="SMAC, MPE; closes gap between policy-gradient and value-based MARL.",
        limitations="Linear decomposition; same expressiveness ceiling as VDN/QMIX-family.",
        connection="Useful template for value-decomposition with policy-gradient training.",
    ),
    dict(
        id="1906.04737", key="papoudakis2019nonstationarity", lane="A",
        title="Dealing with Non-Stationarity in Multi-Agent Deep Reinforcement Learning",
        authors=("Georgios Papoudakis and Filippos Christianos and Arrasy Rahman and "
                 "Stefano V. Albrecht"),
        year=2019, venue="arXiv survey",
        abstract=(
            "Survey of the non-stationarity problem in multi-agent deep RL, covering "
            "centralised training, opponent modelling, meta-learning, communication, "
            "and decentralised approaches."
        ),
        method="Survey — not an algorithm.",
        benchmarks="Coverage of methods through 2019.",
        limitations="Pre-MAPPO/MAT/HAPPO; cites no SMAC successor.",
        connection="Useful reference for framing related work in section 2.",
    ),
    dict(
        id="2006.07869", key="papoudakis2021epymarl", lane="A",
        title="Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks",
        authors=("Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and "
                 "Stefano V. Albrecht"),
        year=2021, venue="NeurIPS Datasets and Benchmarks",
        abstract=(
            "Systematic comparison of independent, centralised-PG, and value-"
            "decomposition MARL methods across a diverse cooperative test suite; "
            "introduces EPyMARL, an extension of PyMARL with more algorithms and "
            "configurations."
        ),
        method="Empirical benchmarking; codebase release.",
        benchmarks="MPE, SMAC, LBF, RWARE; published per-task numbers.",
        limitations="2021 vintage; doesn't include MAT/HAPPO.",
        connection="Likely source of our published numbers to reproduce.",
    ),
    dict(
        id="2210.13708", key="hu2023marllib", lane="A",
        title="MARLlib: A Scalable and Efficient Multi-agent Reinforcement Learning Library",
        authors=("Siyi Hu and Yifan Zhong and Minquan Gao and Weixun Wang and Hao Dong "
                 "and Xiaodan Liang and Zhihui Li and Xiaojun Chang and Yaodong Yang"),
        year=2023, venue="JMLR",
        abstract=(
            "Unified MARL library: standardised env wrappers, agent-level algorithms, "
            "flexible policy mapping, built atop RLlib. Disentangles task properties "
            "from algorithm choice."
        ),
        method="Framework — not an algorithm.",
        benchmarks="Reproduces baselines across MPE, SMAC, GRF, MAMujoco.",
        limitations="RLlib dependency; opinionated config layer.",
        connection="Optional vendor source if we need a vetted MAPPO implementation.",
    ),
    # --------------------------------- Lane B ---------------------------------
    dict(
        id="1707.06347", key="schulman2017ppo", lane="B",
        title="Proximal Policy Optimization Algorithms",
        authors=("John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford "
                 "and Oleg Klimov"),
        year=2017, venue="arXiv preprint",
        abstract=(
            "PPO is a policy-gradient method that alternates rollout collection with "
            "multiple epochs of mini-batch updates on a clipped surrogate objective. "
            "Practical alternative to TRPO with simpler implementation and competitive "
            "sample complexity."
        ),
        method=(
            "Clipped surrogate L^CLIP(θ) = E[min(r_t A, clip(r_t,1-ε,1+ε) A)] with "
            "r_t = π_θ(a|s)/π_old(a|s); k mini-batch epochs per rollout; entropy bonus."
        ),
        benchmarks="MuJoCo locomotion, Atari; matches/beats A2C/TRPO.",
        limitations="Sensitive to many implementation details (advantage normalisation, value clipping).",
        connection="Backbone of both MAPPO/HAPPO and modern RLHF — universal substrate.",
    ),
    dict(
        id="2203.02155", key="ouyang2022instructgpt", lane="B",
        title="Training Language Models to Follow Instructions with Human Feedback",
        authors=("Long Ouyang and Jeff Wu and Xu Jiang and Diogo Almeida and "
                 "Carroll L. Wainwright and Pamela Mishkin and Chong Zhang and "
                 "Sandhini Agarwal and Katarina Slama and Alex Ray and John Schulman "
                 "and Jacob Hilton and Fraser Kelton and Luke Miller and Maddie Simens "
                 "and Amanda Askell and Peter Welinder and Paul Christiano and "
                 "Jan Leike and Ryan Lowe"),
        year=2022, venue="NeurIPS",
        abstract=(
            "InstructGPT: SFT on demonstrations → reward model on rankings → PPO "
            "against the reward model. A 1.3B PPO model is preferred over the 175B "
            "GPT-3 baseline; improves truthfulness and reduces toxicity."
        ),
        method="3-stage RLHF: SFT, RM training, PPO with KL penalty to the SFT policy.",
        benchmarks="API prompt distribution; TruthfulQA, RealToxicityPrompts.",
        limitations="Reward hacking; expensive PPO loop; RM brittleness on out-of-distribution prompts.",
        connection="Defines the PPO-for-LLMs paradigm; substrate for everything in Lane B/C.",
    ),
    dict(
        id="2305.18290", key="rafailov2023dpo", lane="B",
        title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        authors=("Rafael Rafailov and Archit Sharma and Eric Mitchell and Stefano Ermon "
                 "and Christopher D. Manning and Chelsea Finn"),
        year=2023, venue="NeurIPS",
        abstract=(
            "DPO reparameterises the RLHF objective so the optimal policy has a closed "
            "form in terms of the reward; this enables training directly from "
            "preference pairs with a classification loss, eliminating sampling, KL "
            "control, and explicit reward modelling."
        ),
        method=(
            "Loss = -log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)));"
            " y_w preferred over y_l; β = inverse temperature."
        ),
        benchmarks="IMDb sentiment, TL;DR, Anthropic HH; matches or beats PPO RLHF.",
        limitations="Single-step preference assumption; can over-optimise the gap on rare pairs.",
        connection="Lower-compute alternative to PPO for our agent-tuning experiments.",
    ),
    dict(
        id="2402.03300", key="shao2024grpo", lane="B",
        title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        authors=("Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao "
                 "Song and Xiao Bi and Haowei Zhang and Mingchuan Zhang and Y. K. Li "
                 "and Y. Wu and Daya Guo"),
        year=2024, venue="arXiv preprint",
        abstract=(
            "DeepSeekMath 7B continues pre-training on 120B math tokens and is fine-"
            "tuned with Group Relative Policy Optimisation (GRPO), a PPO variant that "
            "drops the value critic in favour of group-relative advantages — cheaper "
            "memory, comparable performance."
        ),
        method=(
            "GRPO: for each prompt sample G completions, compute group-mean reward and "
            "use (r_i - mean(r))/std(r) as advantage; no value network; same clipped "
            "PPO surrogate."
        ),
        benchmarks="MATH 51.7%, 60.9% w/ self-consistency.",
        limitations="Strong only on verifiable-reward domains (math/code); requires G≥2 samples per prompt.",
        connection="Cost-aligned alternative to InstructGPT-style PPO for agent fine-tuning.",
    ),
    dict(
        id="2210.03629", key="yao2022react", lane="B",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        authors=("Shunyu Yao and Jeffrey Zhao and Dian Yu and Nan Du and Izhak Shafran "
                 "and Karthik Narasimhan and Yuan Cao"),
        year=2022, venue="ICLR 2023",
        abstract=(
            "Interleaves chain-of-thought reasoning with environment actions, allowing "
            "an LLM to plan, act, observe, and replan. Improves over CoT-only and "
            "imitation/RL baselines on HotpotQA, Fever, ALFWorld, and WebShop."
        ),
        method=(
            "Few-shot prompted LLM produces Thought/Act/Obs triples in a loop until a "
            "Finish action. No fine-tuning required for the headline results."
        ),
        benchmarks="HotpotQA, Fever, ALFWorld (+34% abs.), WebShop (+10% abs.).",
        limitations="Brittle to prompt format; long contexts; no learned credit assignment.",
        connection="Standard agentic-LLM evaluation pattern; ALFWorld is a candidate Lane B benchmark.",
    ),
    dict(
        id="2303.11366", key="shinn2023reflexion", lane="B",
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        authors=("Noah Shinn and Federico Cassano and Edward Berman and Ashwin Gopinath "
                 "and Karthik Narasimhan and Shunyu Yao"),
        year=2023, venue="NeurIPS",
        abstract=(
            "Replaces gradient updates with linguistic self-reflection: after each "
            "failed trial, the agent writes a self-critique that is appended to "
            "context for the next trial, achieving multi-trial improvement without "
            "weight updates."
        ),
        method=(
            "Actor LLM → environment trajectory → evaluator computes scalar reward → "
            "self-reflection generator writes natural-language critique → memory "
            "buffer; repeated for k trials."
        ),
        benchmarks="ALFWorld, HotpotQA, HumanEval/MBPP; large gains over ReAct baselines.",
        limitations="Relies on the LLM's introspective capacity; weak on novel tool surfaces.",
        connection="Sample-efficient inference-time RL alternative; orthogonal to gradient methods.",
    ),
    dict(
        id="2308.08155", key="wu2023autogen", lane="B",
        title="AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        authors=("Qingyun Wu and Gagan Bansal and Jieyu Zhang and Yiran Wu and Beibin "
                 "Li and Erkang Zhu and Li Jiang and Xiaoyun Zhang and Shaokun Zhang "
                 "and Jiale Liu and Ahmed Hassan Awadallah and Ryen W. White and "
                 "Doug Burger and Chi Wang"),
        year=2023, venue="arXiv preprint",
        abstract=(
            "Open framework for building LLM apps as conversations among customisable "
            "agents (LLM, tool, human). Demonstrates wide applicability across math, "
            "coding, QA, decision-making."
        ),
        method="Framework: ConversableAgent abstraction with chat-history-based message passing.",
        benchmarks="Case studies; no headline RL training.",
        limitations="No learning loop; quality bottlenecked by underlying LLM.",
        connection="Reference architecture for multi-agent LLM systems we'd train at scale.",
    ),
    dict(
        id="2308.00352", key="hong2023metagpt", lane="B",
        title="MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework",
        authors=("Sirui Hong and Mingchen Zhuge and Jiaqi Chen and Xiawu Zheng and Yuheng "
                 "Cheng and Ceyao Zhang and Jinlin Wang and Zili Wang and "
                 "Steven Ka Shing Yau and Zijuan Lin and Liyang Zhou and Chenyu Ran and "
                 "Lingfeng Xiao and Chenglin Wu and Jürgen Schmidhuber"),
        year=2023, venue="ICLR 2024",
        abstract=(
            "Encodes Standardised Operating Procedures (SOPs) into prompt sequences to "
            "decompose software-engineering tasks across role-specialised LLM agents "
            "(PM/Architect/Engineer/QA), reducing cascading hallucinations."
        ),
        method="SOP-prompted role agents communicating via shared memory; assembly-line workflow.",
        benchmarks="HumanEval, MBPP, software engineering tasks; outperforms chat-only baselines.",
        limitations="Hand-engineered workflow; no end-to-end learning.",
        connection="Multi-agent LLM coordination without RL — useful contrast point.",
    ),
    dict(
        id="2303.17760", key="li2023camel", lane="B",
        title="CAMEL: Communicative Agents for \"Mind\" Exploration of Large Language Model Society",
        authors=("Guohao Li and Hasan Abed Al Kader Hammoud and Hani Itani and "
                 "Dmitrii Khizbullin and Bernard Ghanem"),
        year=2023, venue="NeurIPS",
        abstract=(
            "Role-playing framework that uses inception prompting to keep two LLM "
            "agents on task; generates large-scale instruction data through agent-"
            "to-agent dialogue."
        ),
        method="Two-agent role-play with inception prompt; transcripts collected as data.",
        benchmarks="Conversational datasets; downstream fine-tuning gains.",
        limitations="Role-play heuristics; no quantitative cooperation benchmark.",
        connection="Multi-agent LLM data generation precedent; related to self-play.",
    ),
    dict(
        id="2302.04761", key="schick2023toolformer", lane="B",
        title="Toolformer: Language Models Can Teach Themselves to Use Tools",
        authors=("Timo Schick and Jane Dwivedi-Yu and Roberto Dessì and Roberta Raileanu "
                 "and Maria Lomeli and Luke Zettlemoyer and Nicola Cancedda and "
                 "Thomas Scialom"),
        year=2023, venue="NeurIPS",
        abstract=(
            "Self-supervised tool-use: the model proposes API calls inside generated "
            "text, executes them, and keeps the call only if the answer improves the "
            "downstream language-modelling loss."
        ),
        method="Sample API-call insertions; filter by perplexity gain; fine-tune on retained calls.",
        benchmarks="Calculator, QA, search, translation tools; improves zero-shot QA.",
        limitations="Filtering proxy is brittle; no multi-step tool composition.",
        connection="Earliest large-scale self-supervised tool-RL signal.",
    ),
    dict(
        id="2305.16291", key="wang2023voyager", lane="B",
        title="Voyager: An Open-Ended Embodied Agent with Large Language Models",
        authors=("Guanzhi Wang and Yuqi Xie and Yunfan Jiang and Ajay Mandlekar and "
                 "Chaowei Xiao and Yuke Zhu and Linxi Fan and Anima Anandkumar"),
        year=2023, venue="TMLR",
        abstract=(
            "GPT-4-powered lifelong-learning agent in Minecraft with automatic "
            "curriculum, skill library of executable code, and iterative prompting "
            "with environmental feedback; outperforms prior SOTA on item collection "
            "and tech-tree progress."
        ),
        method="Three components: curriculum generator, skill code library, self-verifier.",
        benchmarks="Minecraft; 3.3× items, 2.3× distance, 15.3× faster milestones.",
        limitations="Closed-source GPT-4 dependency; Minecraft-specific.",
        connection="Open-ended embodied agent setup, useful for grounding scaling claims.",
    ),
    dict(
        id="2308.03688", key="liu2023agentbench", lane="B",
        title="AgentBench: Evaluating LLMs as Agents",
        authors=("Xiao Liu and Hao Yu and Hanchen Zhang and Yifan Xu and Xuanyu Lei and "
                 "Hanyu Lai and Yu Gu and Hangliang Ding and Kaiwen Men and Kejuan Yang "
                 "and Shudan Zhang and Xiang Deng and Aohan Zeng and Zhengxiao Du and "
                 "Chenhui Zhang and Sheng Shen and Tianjun Zhang and Yu Su and Huan Sun "
                 "and Minlie Huang and Yuxiao Dong and Jie Tang"),
        year=2023, venue="ICLR 2024",
        abstract=(
            "AgentBench: 8-environment evaluation suite for LLM-as-agent across OS, "
            "DB, web shopping, knowledge graphs, etc. API LLMs lead; open-source <70B "
            "models trail in long-horizon reasoning."
        ),
        method="Benchmark + standardised harness.",
        benchmarks="GPT-4 dominates; long-horizon reasoning is the dominant failure mode.",
        limitations="Heavy compute; closed-tool surfaces.",
        connection="Reference suite; out of scope as a training target for our budget.",
    ),
    dict(
        id="2307.13854", key="zhou2023webarena", lane="B",
        title="WebArena: A Realistic Web Environment for Building Autonomous Agents",
        authors=("Shuyan Zhou and Frank F. Xu and Hao Zhu and Xuhui Zhou and Robert Lo "
                 "and Abishek Sridhar and Xianyi Cheng and Tianyue Ou and Yonatan Bisk "
                 "and Daniel Fried and Uri Alon and Graham Neubig"),
        year=2023, venue="ICLR 2024",
        abstract=(
            "Realistic web environment for autonomous agents across e-commerce, social "
            "forum, dev, and CMS. GPT-4 agent at 14.4% success vs 78.2% human."
        ),
        method="Benchmark — hosted reproducible web apps with task suite.",
        benchmarks="GPT-4 14.4% success.",
        limitations="Requires hosting; long horizons.",
        connection="Aspirational LLM-agent target; not within our budget.",
    ),
    dict(
        id="2305.20050", key="lightman2023letsverify", lane="B",
        title="Let's Verify Step by Step",
        authors=("Hunter Lightman and Vineet Kosaraju and Yura Burda and Harri Edwards "
                 "and Bowen Baker and Teddy Lee and Jan Leike and John Schulman and "
                 "Ilya Sutskever and Karl Cobbe"),
        year=2023, venue="ICLR 2024",
        abstract=(
            "Process-based reward models (PRMs) that supervise each reasoning step "
            "outperform outcome-only reward models on MATH; releases PRM800K, the "
            "complete dataset of 800k step-level human labels."
        ),
        method="Train PRM as binary classifier per step; use best-of-N or RL with the PRM.",
        benchmarks="78% on a representative MATH subset.",
        limitations="Heavy annotation cost; only verifiable-domain rewards.",
        connection="Process-level credit assignment — analogous to per-agent credit in MARL.",
    ),
    dict(
        id="2305.14325", key="du2023multiagentdebate", lane="B",
        title="Improving Factuality and Reasoning in Language Models through Multiagent Debate",
        authors=("Yilun Du and Shuang Li and Antonio Torralba and Joshua B. Tenenbaum "
                 "and Igor Mordatch"),
        year=2023, venue="ICML 2024",
        abstract=(
            "Multiple LLM instances propose answers, then critique each other across "
            "rounds; consensus answers are more factual and improve reasoning."
        ),
        method="K agents × R rounds of self-critique; majority/last-round answer used.",
        benchmarks="Math, biographies, GSM8K; gains over single-agent CoT.",
        limitations="K·R-fold inference cost; no learning.",
        connection="Inference-time multi-agent LLM scaling — relevant to distributed rollouts.",
    ),
    dict(
        id="2401.01335", key="chen2024spin", lane="B",
        title="Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models",
        authors=("Zixiang Chen and Yihe Deng and Huizhuo Yuan and Kaixuan Ji and "
                 "Quanquan Gu"),
        year=2024, venue="ICML",
        abstract=(
            "SPIN trains a single LLM iteratively to distinguish its own samples from "
            "human-annotated data; the resulting policy converges to the target "
            "distribution. Outperforms DPO with GPT-4-labelled preferences in some "
            "settings."
        ),
        method=(
            "Iter t: generate y' from π_{t-1}, optimise π_t with DPO-style loss using "
            "(human y_target ≻ y') pairs; no external preferences."
        ),
        benchmarks="HuggingFace Open LLM Leaderboard, MT-Bench.",
        limitations="Self-play can collapse; requires high-quality SFT starting point.",
        connection="Self-play LLM training: relevant to single-policy MARL framings.",
    ),
    dict(
        id="2310.12823", key="zeng2023agenttuning", lane="B",
        title="AgentTuning: Enabling Generalized Agent Abilities for LLMs",
        authors=("Aohan Zeng and Mingdao Liu and Rui Lu and Bowen Wang and Xiao Liu and "
                 "Yuxiao Dong and Jie Tang"),
        year=2023, venue="arXiv preprint",
        abstract=(
            "AgentTuning: high-quality agent-interaction dataset (AgentInstruct) mixed "
            "with general SFT data. AgentLM-70B matches GPT-3.5-turbo on unseen agent "
            "tasks while preserving general LM capability."
        ),
        method="SFT on 50/50 mix of AgentInstruct trajectories and general instructions.",
        benchmarks="AgentBench unseen tasks; preserves MMLU/HumanEval.",
        limitations="Closed-loop quality bottleneck of GPT-4-generated trajectories.",
        connection="Reference for agent fine-tuning data strategy.",
    ),
    dict(
        id="2010.03768", key="shridhar2020alfworld", lane="B",
        title="ALFWorld: Aligning Text and Embodied Environments for Interactive Learning",
        authors=("Mohit Shridhar and Xingdi Yuan and Marc-Alexandre Côté and Yonatan "
                 "Bisk and Adam Trischler and Matthew Hausknecht"),
        year=2020, venue="ICLR 2021",
        abstract=(
            "Aligns ALFRED's embodied household tasks with TextWorld text descriptions; "
            "agents trained in text generalise zero-shot to grounded action."
        ),
        method="Parallel text+vision environments; BUTLER agent with separate policies.",
        benchmarks="Text → vision zero-shot transfer; standard agentic benchmark.",
        limitations="Discrete action surface; relatively short horizons.",
        connection="Candidate lightweight Lane-B benchmark.",
    ),
    dict(
        id="1810.08272", key="chevalierboisvert2018babyai", lane="B",
        title="BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning",
        authors=("Maxime Chevalier-Boisvert and Dzmitry Bahdanau and Salem Lahlou and "
                 "Lucas Willems and Chitwan Saharia and Thien Huu Nguyen and "
                 "Yoshua Bengio"),
        year=2018, venue="ICLR 2019",
        abstract=(
            "19-level grid-world platform for instruction-following; expert agent "
            "stand-in for a human teacher; quantifies sample inefficiency of current "
            "deep RL on a compositional synthetic language."
        ),
        method="MiniGrid-based environments + procedural language generation.",
        benchmarks="Sample-efficiency baselines per level.",
        limitations="Compositional but synthetic language; small action surface.",
        connection="Lightweight grounded-language benchmark; small compute footprint.",
    ),
    # --------------------------------- Lane C ---------------------------------
    dict(
        id="1802.01561", key="espeholt2018impala", lane="C",
        title="IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures",
        authors=("Lasse Espeholt and Hubert Soyer and Rémi Munos and Karen Simonyan and "
                 "Volodymyr Mnih and Tom Ward and Yotam Doron and Vlad Firoiu and "
                 "Tim Harley and Iain Dunning and Shane Legg and Koray Kavukcuoglu"),
        year=2018, venue="ICML",
        abstract=(
            "Decouples actors from a centralised learner; introduces V-trace off-policy "
            "correction for the lag between actor and learner policies; trains a single "
            "agent across DMLab-30 and Atari-57 with positive transfer."
        ),
        method=(
            "Many actors generate trajectories with stale params; learner does on-GPU "
            "updates with V-trace corrected targets v_s = V(s) + Σγ^{t-s} ρ_t δ_t."
        ),
        benchmarks="DMLab-30 multi-task; Atari-57; throughput >250k FPS.",
        limitations="V-trace truncates importance weights; not Pareto-optimal for very off-policy data.",
        connection="Reference architecture for distributed actor-learner — relevant to scaling MARL rollouts.",
    ),
    dict(
        id="1910.06591", key="espeholt2020seedrl", lane="C",
        title="SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference",
        authors=("Lasse Espeholt and Raphaël Marinier and Piotr Stanczyk and Ke Wang "
                 "and Marcin Michalski"),
        year=2019, venue="ICLR 2020",
        abstract=(
            "Centralised inference on accelerators with thin observation-only actors. "
            "Reaches millions of FPS; matches SOTA on Atari-57 3× faster wall-clock and "
            "cuts experiment cost 40-80%."
        ),
        method="Actor sends obs to a central inference server; server returns actions; gRPC streams.",
        benchmarks="Atari-57, DMLab, GRF.",
        limitations="Network is the bottleneck for very small per-step compute.",
        connection="Centralised-inference design directly relevant to LLM-agent rollouts.",
    ),
    dict(
        id="1803.00933", key="horgan2018apex", lane="C",
        title="Distributed Prioritized Experience Replay",
        authors=("Dan Horgan and John Quan and David Budden and Gabriel Barth-Maron and "
                 "Matteo Hessel and Hado van Hasselt and David Silver"),
        year=2018, venue="ICLR",
        abstract=(
            "Ape-X separates many CPU actors with their own replay priorities from a "
            "single GPU learner sampling from shared prioritised replay; massively "
            "scales DQN and DPG variants."
        ),
        method="N actors compute initial TD priorities; central PER buffer; one learner samples.",
        benchmarks="Atari — SOTA in much less wall-clock.",
        limitations="Off-policy + stale priorities can drift; large replay buffer memory.",
        connection="Asynchronous-actor design pattern reused throughout MARL.",
    ),
    dict(
        id="1712.09381", key="liang2017rllib", lane="C",
        title="RLlib: Abstractions for Distributed Reinforcement Learning",
        authors=("Eric Liang and Richard Liaw and Philipp Moritz and Robert Nishihara "
                 "and Roy Fox and Ken Goldberg and Joseph E. Gonzalez and "
                 "Michael I. Jordan and Ion Stoica"),
        year=2017, venue="ICML 2018",
        abstract=(
            "Builds RL on top of Ray with hierarchical control; encapsulates parallelism "
            "and resource needs inside small tasks. Provides composable primitives for "
            "policy evaluation, optimisation, and exploration."
        ),
        method="Ray actor model + abstract Trainer/Policy/RolloutWorker classes.",
        benchmarks="A3C/IMPALA/PPO scale-out demos.",
        limitations="Opinionated API; debugging across Ray actors hurts.",
        connection="Backbone of MARLlib; relevant for distributed-MARL infra positioning.",
    ),
    dict(
        id="2006.00979", key="hoffman2020acme", lane="C",
        title="Acme: A Research Framework for Distributed Reinforcement Learning",
        authors=("Matthew W. Hoffman and many others"),
        year=2020, venue="arXiv preprint",
        abstract=(
            "Acme: modular framework for constructing distributed RL agents with a "
            "small set of clean abstractions; reference implementations of D4PG, "
            "MPO, IMPALA, R2D2, etc."
        ),
        method="Framework — Reverb-based replay, Launchpad for distribution.",
        benchmarks="Atari, DMC, DMLab baselines.",
        limitations="DeepMind-internal flavour of distribution; Reverb dependency.",
        connection="Alternative to RLlib for serious distributed runs.",
    ),
    dict(
        id="2309.06180", key="kwon2023vllm", lane="C",
        title="Efficient Memory Management for Large Language Model Serving with PagedAttention",
        authors=("Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and "
                 "Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang "
                 "and Ion Stoica"),
        year=2023, venue="SOSP",
        abstract=(
            "PagedAttention manages the KV cache as fixed-size pages, eliminating "
            "fragmentation and enabling cache sharing across requests. vLLM achieves "
            "2-4× higher throughput than FasterTransformer/Orca."
        ),
        method="Virtual-memory-like paging over KV blocks; copy-on-write for shared prefixes.",
        benchmarks="LLaMA-7B/13B/70B; ShareGPT prompts.",
        limitations="Block size hyperparameter; integration cost with frameworks.",
        connection="Standard inference engine for RLHF rollouts (used in HybridFlow/OpenRLHF).",
    ),
    dict(
        id="2409.19256", key="sheng2024hybridflow", lane="C",
        title="HybridFlow: A Flexible and Efficient RLHF Framework",
        authors=("Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and "
                 "Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu"),
        year=2024, venue="EuroSys 2025",
        abstract=(
            "Combines single- and multi-controller paradigms for RLHF dataflows; "
            "introduces 3D-HybridEngine for actor model resharding between rollout and "
            "training; 1.53-20.57× throughput vs SOTA."
        ),
        method="Hierarchical APIs; placement strategy split across actor/ref/critic/reward roles.",
        benchmarks="LLaMA-7B/13B/70B RLHF.",
        limitations="Complex placement decisions; framework-specific.",
        connection="Most directly relevant infra paper to a distributed-LLM-RL contribution.",
    ),
    dict(
        id="2405.11143", key="hu2024openrlhf", lane="C",
        title="OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework",
        authors=("Jian Hu and Xibin Wu and Wei Shen and Jason Klein Liu and Zilin Zhu "
                 "and Weixun Wang and Songlin Jiang and Haoran Wang and Hao Chen and "
                 "Bin Chen and Weikai Fang and Xianyu and Yu Cao and Haotian Xu and "
                 "Yiming Liu"),
        year=2024, venue="arXiv preprint",
        abstract=(
            "Open-source RLHF framework atop Ray + vLLM + DeepSpeed + HF Transformers; "
            "1.22-1.68× speedup across model sizes vs existing frameworks."
        ),
        method="Disaggregated actor/critic/reference/rollout placement; vLLM rollouts.",
        benchmarks="LLaMA-7B–70B RLHF.",
        limitations="Inherits vLLM/Ray operational complexity.",
        connection="Reference architecture for an OSS RL pipeline.",
    ),
    dict(
        id="2308.01320", key="yao2023deepspeedchat", lane="C",
        title="DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales",
        authors=("Zhewei Yao and Reza Yazdani Aminabadi and Olatunji Ruwase and Samyam "
                 "Rajbhandari and Xiaoxia Wu and Ammar Ahmad Awan and Jeff Rasley and "
                 "Minjia Zhang and Conglong Li and Connor Holmes and Zhongzhu Zhou and "
                 "Michael Wyatt and Molly Smith and Lev Kurilenko and Heyang Qin and "
                 "Masahiro Tanaka and Shuai Che and Shuaiwen Leon Song and Yuxiong He"),
        year=2023, venue="arXiv preprint",
        abstract=(
            "End-to-end RLHF pipeline built on DeepSpeed: SFT, RM, PPO at hundreds of "
            "billions of parameters with ZeRO + hybrid engine for rollout/training."
        ),
        method="Three-stage RLHF; hybrid inference/training engine; ZeRO-Offload.",
        benchmarks="Throughput tables vs HF-based pipelines.",
        limitations="DeepSpeed-specific; less flexible composition than Ray-based stacks.",
        connection="Industrial reference for scaling RLHF.",
    ),
    dict(
        id="2111.08819", key="huang2022cleanrl", lane="C",
        title="CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms",
        authors=("Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and "
                 "Jeff Braga"),
        year=2021, venue="JMLR Open Source Software 2022",
        abstract=(
            "Each algorithm in one readable file with production-grade logging and "
            "experiment tracking; validated across many environments."
        ),
        method="Single-file impl style; wandb/tb integration.",
        benchmarks="Atari/MuJoCo/Procgen reproduction tables.",
        limitations="Single-file style does not scale to large algorithm families.",
        connection="Reference implementations we will likely vendor (PPO single file).",
    ),
    dict(
        id="2008.08932", key="terry2020supersuit", lane="C",
        title="SuperSuit: Simple Microwrappers for Reinforcement Learning Environments",
        authors="J. K. Terry and Benjamin Black and Ananth Hari",
        year=2020, venue="arXiv preprint",
        abstract=(
            "Lightweight, composable env wrappers compatible with Gym and PettingZoo; "
            "standardises pre-processing and vectorisation."
        ),
        method="Wrapper library — concat/observation/action/space modifiers.",
        benchmarks="N/A.",
        limitations="Surface-level utilities; opinionated defaults.",
        connection="Used in our env stack.",
    ),
    dict(
        id="2009.14471", key="terry2020pettingzoo", lane="C",
        title="PettingZoo: Gym for Multi-Agent Reinforcement Learning",
        authors=("J. K. Terry and Benjamin Black and Nathaniel Grammel and Mario "
                 "Jayakumar and Ananth Hari and Ryan Sullivan and Luis Santos and "
                 "Rodrigo Perez and Caroline Horsch and Clemens Dieffendahl and "
                 "Niall L. Williams and Yashas Lokesh and Praveen Ravi"),
        year=2020, venue="NeurIPS 2021",
        abstract=(
            "Unified MARL env API with the AEC (Agent-Environment-Cycle) model; broad "
            "library of cooperative, mixed, and competitive envs."
        ),
        method="AEC API supplementing parallel API.",
        benchmarks="MPE, Butterfly, SISL, Atari multi-agent.",
        limitations="MPE moved to mpe2 in v1.25+; minor API churn.",
        connection="Our env-API standard.",
    ),
    dict(
        id="2009.14794", key="choromanski2021performers", lane="C",
        title="Rethinking Attention with Performers",
        authors=("Krzysztof Choromanski and Valerii Likhosherstov and David Dohan and "
                 "Xingyou Song and Andreea Gane and Tamas Sarlos and Peter Hawkins and "
                 "Jared Davis and Afroz Mohiuddin and Lukasz Kaiser and David Belanger "
                 "and Lucy Colwell and Adrian Weller"),
        year=2020, venue="ICLR 2021",
        abstract=(
            "Performers approximate full-rank softmax attention in linear time and "
            "space via random feature maps (FAVOR+), enabling long-context attention "
            "where vanilla transformers are infeasible."
        ),
        method="Positive orthogonal random features approximating softmax kernel.",
        benchmarks="Pixel modelling, LM, protein sequences.",
        limitations="Approximation error trade-off; less effective at small N.",
        connection="Possible method for scaling agent-attention to large N without O(N²) cost.",
    ),
    # --------------------------------- Lane D ---------------------------------
    dict(
        id="1802.05438", key="yang2018meanfield", lane="D",
        title="Mean Field Multi-Agent Reinforcement Learning",
        authors=("Yaodong Yang and Rui Luo and Minne Li and Ming Zhou and Weinan Zhang "
                 "and Jun Wang"),
        year=2018, venue="ICML",
        abstract=(
            "Approximates many-agent interactions by the mean effect of "
            "neighbours/population; develops MF-Q and MF-AC; convergence analysed via "
            "Nash equilibrium of the induced game."
        ),
        method=(
            "Replace joint-action arg in Q(s,a) by Q(s,a_i, ā_{-i}) with ā being mean "
            "neighbour action; iterate fixed-point updates."
        ),
        benchmarks="Gaussian squeeze, Ising model, battle games (100s of agents).",
        limitations="Mean-field assumption breaks down when local heterogeneity matters.",
        connection="Reference for the large-N regime our scaling experiments should claim.",
    ),
    dict(
        id="1711.09846", key="jaderberg2017pbt", lane="D",
        title="Population Based Training of Neural Networks",
        authors=("Max Jaderberg and Valentin Dalibard and Simon Osindero and Wojciech "
                 "M. Czarnecki and Jeff Donahue and Ali Razavi and Oriol Vinyals and "
                 "Tim Green and Iain Dunning and Karen Simonyan and Chrisantha Fernando "
                 "and Koray Kavukcuoglu"),
        year=2017, venue="arXiv preprint",
        abstract=(
            "PBT jointly optimises population members and hyperparameters via "
            "asynchronous exploit/explore: weak agents copy weights+hyperparams from "
            "strong ones and perturb them. Discovers schedules rather than fixed "
            "settings."
        ),
        method="Periodic ready/exploit/explore cycle; truncation selection.",
        benchmarks="UNREAL/DRL agents, NMT, GANs.",
        limitations="Compute-hungry; weight-copy can disrupt training.",
        connection="Population-based MARL training pattern; relevant to scaling.",
    ),
    dict(
        id="1902.00506", key="bard2019hanabi", lane="D",
        title="The Hanabi Challenge: A New Frontier for AI Research",
        authors=("Nolan Bard and Jakob N. Foerster and Sarath Chandar and Neil Burch "
                 "and Marc Lanctot and H. Francis Song and Emilio Parisotto and "
                 "Vincent Dumoulin and Subhodeep Moitra and Edward Hughes and "
                 "Iain Dunning and Shibl Mourad and Hugo Larochelle and "
                 "Marc G. Bellemare and Michael Bowling"),
        year=2019, venue="Artificial Intelligence Journal",
        abstract=(
            "Proposes Hanabi as a benchmark combining cooperation, theory-of-mind, "
            "and imperfect information. Provides the Hanabi Learning Environment and "
            "baseline evaluations."
        ),
        method="Benchmark + open-source environment.",
        benchmarks="Self-play and ad-hoc team play across multiple agents.",
        limitations="Discrete card game; specific to imperfect-information cooperation.",
        connection="Lightweight cooperative benchmark with strong ToM signal.",
    ),
    dict(
        id="2107.06857", key="leibo2021meltingpot", lane="D",
        title="Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot",
        authors=("Joel Z. Leibo and Edgar Duéñez-Guzmán and Alexander Sasha Vezhnevets "
                 "and John P. Agapiou and Peter Sunehag and Raphael Koster and "
                 "Jayd Matyas and Charles Beattie and Igor Mordatch and Thore Graepel"),
        year=2021, venue="ICML",
        abstract=(
            "80+ MARL test scenarios across social dilemmas, reciprocity, resource "
            "sharing; uses RL-trained background bots so evaluation environment is "
            "another learned policy."
        ),
        method="Substrates + bots + scenarios; standard CTDE training in scenarios.",
        benchmarks="Per-substrate baseline comparisons.",
        limitations="Heavier compute than MPE/Hanabi.",
        connection="Reference for mixed-motive evaluation; not in budget as primary benchmark.",
    ),
    dict(
        id="1910.05789", key="carroll2019overcooked", lane="D",
        title="On the Utility of Learning about Humans for Human-AI Coordination",
        authors=("Micah Carroll and Rohin Shah and Mark K. Ho and Thomas L. Griffiths "
                 "and Sanjit A. Seshia and Pieter Abbeel and Anca Dragan"),
        year=2019, venue="NeurIPS",
        abstract=(
            "Self-play agents excel at self-play but fail with human partners on "
            "Overcooked; agents trained against learned human models coordinate "
            "better. User study confirms."
        ),
        method="Overcooked-AI env + behaviour-cloning human models + SP/PBT baselines.",
        benchmarks="Overcooked layouts; self-play vs human-aware training.",
        limitations="Specific to two-agent cooperative cooking; behaviour cloning of humans is noisy.",
        connection="Two-agent cooperative benchmark with low compute footprint.",
    ),
    dict(
        id="1912.06680", key="berner2019openai5", lane="D",
        title="Dota 2 with Large Scale Deep Reinforcement Learning",
        authors=("Christopher Berner and Greg Brockman and Brooke Chan and Vicki Cheung "
                 "and Przemysław Dębiak and Christy Dennison and David Farhi and Quirin "
                 "Fischer and Shariq Hashme and Chris Hesse and Rafal Józefowicz and "
                 "Scott Gray and Catherine Olsson and Jakub Pachocki and Michael "
                 "Petrov and Henrique P. d. O. Pinto and Jonathan Raiman and "
                 "Tim Salimans and Jeremy Schlatter and Jonas Schneider and Szymon "
                 "Sidor and Ilya Sutskever and Jie Tang and Filip Wolski and Susan Zhang"),
        year=2019, venue="arXiv preprint",
        abstract=(
            "OpenAI Five defeats the Dota 2 world champion via PPO + LSTM at massive "
            "scale (~2M frames/2s for 10 months) with a custom distributed system "
            "supporting continual training."
        ),
        method="Distributed PPO on team self-play; surgery for arch changes mid-training.",
        benchmarks="Dota 2 vs Team OG; superhuman.",
        limitations="Compute cost is industrial; many surgical re-warming events.",
        connection="Anchor for scaling claims and continual training relevance.",
    ),
    dict(
        id="1909.07528", key="baker2020hideandseek", lane="D",
        title="Emergent Tool Use From Multi-Agent Autocurricula",
        authors=("Bowen Baker and Ingmar Kanitscheider and Todor Markov and Yi Wu and "
                 "Glenn Powell and Bob McGrew and Igor Mordatch"),
        year=2019, venue="ICLR 2020",
        abstract=(
            "Multi-agent hide-and-seek with standard RL produces an autocurriculum "
            "with six emergent strategy phases (chasing, blocking, box use, ramp use, "
            "ramp defence, box surfing). Suggests competition + complex env > "
            "intrinsic-motivation baselines."
        ),
        method="Self-play PPO at scale on a Bullet-based physics arena.",
        benchmarks="Custom; transfer evaluations.",
        limitations="Compute-heavy; closed-source env.",
        connection="Motivates open-ended large-scale MARL — useful framing.",
    ),
]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def to_bib(entry: dict) -> str:
    return textwrap.dedent(f"""\
        @article{{{entry['key']},
          title         = {{{entry['title']}}},
          author        = {{{entry['authors']}}},
          year          = {{{entry['year']}}},
          journal       = {{{entry.get('venue', 'arXiv preprint')}}},
          eprint        = {{{entry['id']}}},
          archivePrefix = {{arXiv}},
          primaryClass  = {{cs.LG}},
          url           = {{https://arxiv.org/abs/{entry['id']}}}
        }}
        """)


def to_note(entry: dict) -> str:
    return textwrap.dedent(f"""\
        ---
        arxiv_id: {entry['id']}
        key: {entry['key']}
        lane: {entry['lane']}
        title: {entry['title']}
        year: {entry['year']}
        ---

        # {entry['title']}

        **Key:** `{entry['key']}` — arXiv:{entry['id']} ({entry['year']}, {entry.get('venue','arXiv preprint')}).

        **Authors:** {entry['authors']}

        ## Abstract (verbatim or condensed from arXiv)

        {entry['abstract']}

        ## Method (5 lines)

        {entry['method']}

        ## Benchmarks + headline numbers

        {entry['benchmarks']}

        ## Limitations / open problems flagged by authors

        {entry['limitations']}

        ## Connection to MARL-IoTP / target team

        {entry['connection'] or '(no direct connection)'}
        """)


def main() -> None:
    bib_path = LIT / "references.bib"
    with bib_path.open("w") as f:
        f.write("% Generated from scripts/build_literature.py — do not hand-edit.\n")
        f.write(f"% {len(PAPERS)} entries.\n\n")
        for e in PAPERS:
            f.write(to_bib(e))
            f.write("\n")

    for e in PAPERS:
        (NOTES / f"{e['id']}.md").write_text(to_note(e))

    keys = [e["key"] for e in PAPERS]
    lanes = {l: sum(1 for e in PAPERS if e["lane"] == l) for l in "ABCD"}
    print(json.dumps({"n": len(PAPERS), "lanes": lanes, "unique_keys": len(set(keys))}))


if __name__ == "__main__":
    main()
