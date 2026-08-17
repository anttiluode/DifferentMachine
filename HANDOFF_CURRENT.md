# DifferentMachine — CURRENT HANDOFF

**Status:** Gate 0/0b/1A/1B/1C/1D/1E-A/1F-A are executable mechanism receipts. There is still **no full matched architecture result and no novelty claim**.

## One-line thesis

> **The size of the represented machine need not equal the amount of machine executed for each event.**

## Current machine

```text
small shared developmental genome theta
        |
        v
geometry / event substrate creates local encounters
        |
        v
C(t) persistent sparse structure
z(t) persistent local activity / phase state
A(t) addressed / expanding active frontier
m(t) receiver-specific outward consequence
e(t) local eligibility left by actual signal transport
        |
        +-- delayed task consequence stabilizes / weakens used structure
        `-- weak structure prunes; local encounters can regrow it
```

The newest distinction is now explicit:

```text
genotype  != acquired graph phenotype
```

The inherited object can be constant-size even though the acquired phenotype and represented capacity scale with `N`.

---

## Gate ladder

### Gate 0 — same machine, different execution

`EventMachine` lazily updates only addressed state; `ClockedMachine` explicitly sweeps everything.

```text
4096 contacts
64 branches
20,000 ticks
382 events
max candidate mismatch       1.11e-16
max receiver mismatch        4.16e-17
clocked private touches      83,200,000
lazy private touches         764
logical touch ratio          108,900x
```

Receipt only: quiet persistent state need not be recomputed every global tick.

### Gate 0b — receiver-relative promotion

Under a hard 20% promotion budget, a persistent receiver/branch relationship score beats pure candidate magnitude in the deterministic toy. Relationship weights are privileged, so this is not a result.

### Gate 1A — coupled frontier

Sparse coupled cable graph + best-first active frontier. Receiver relation beats generic/magnitude scoring across nonzero coupling, but strong conditional-compute baselines are still missing.

### Gate 1B — capacity vs per-event work

Question:

```text
Can represented capacity N grow much faster than required per-event work W(N, eps)?
```

```text
bounded degree=3       W ~ N^0.001
sqrt(N) degree          W ~ N^0.447
N/8 degree              W ~ N^1.011
global clock reference  W ~ N^1.000
```

Central constraint:

> **Relevance must be locally discoverable.**

### Gate 1C — learned locality

Receiver-labelled co-use is compiled offline into degree-4 overlays. With conflicting receiver neighborhoods:

```text
N      pooled work   receiver-conditioned work   random work
64          137.0                 30.0               207.1
128         285.5                 30.0               399.9
256         586.2                 30.0               801.2
512        1230.2                 30.0              1603.3
```

Receiver-specific locality stays flat while pooled/random approaches linear. Still associative graph learning territory.

### Gate 1D — bounded online plasticity

Removes offline compile/count-table cheat. Each pair event touches only two degree-4 rows. After a hidden regime change:

```text
steady learned relationship       W ~ N^0.000
immediately after change          W ~ N^0.975
after 48 new events/node          W ~ N^0.004
```

Interpretation:

> **Repeated interaction can amortize future computation into persistent local structure. When the world changes, computation/search spikes, and local plasticity can rebuild the cheap route.**

Big cheat: plasticity still receives `(receiver, i, j)` co-use.

### Gate 1E-A — task consequence replaces useful-pair teacher

Removes the explicit useful-pair teaching event.

```text
ordinary receiver query
+ edge ablation query
+ candidate perturbation query
        |
        v
flow delta eligibility
        ... delayed ...
correct / wrong scalar outcome
```

Two-seed local receipt:

```text
N      initial   learned-old   shift0   recovered
64       .653        .846       .616      .874
128      .640        .874       .617      .886
256      .637        .879       .618      .887
```

Controls at N=128:

```text
main flow x outcome   recovered .886
outcome shuffled                .602
flow only                       .614
reward only                     .656
frozen                          .639
```

But Gate 1E-A still has two ugly cheats:

1. a hand-prepared 12-candidate proposal scaffold;
2. about 89 receiver-query operations/task event because eligibility is measured with counterfactual probes.

### Gate 1F-A — GENOME, NOT GRAPH

Gate 1F-A attacks both of those cheats.

New files:

```text
different_machine/development.py
experiments/gate1f_genome_not_graph.py
tests/test_development.py
docs/GATE1F_GENOME_NOT_GRAPH.md
```

#### Inherited object

A deterministic mutation-only GA evolves **10 shared float64 genes = 80 bytes**.

The genome contains no node ids and no per-edge values. Every node runs the same decoded local rule controlling:

```text
distance preference
recent-activity similarity
optional phase similarity
growth bias
new edge strength
local decay
delayed-credit learning rate
prune threshold
activity-trace decay
activity-driven phase drift
```

Canonical evolution:

```text
training N             48
training world seed    101
evolution seed         7
population             14
generations             9
fitness                 0.680796
```

`python experiments/gate1f_genome_not_graph.py --evolve` exactly reproduced the banked raw genome locally.

#### Candidates are geometry, not a supplied list

Nodes live at constant density in a 2-D substrate. A fixed-radius spatial hash exposes only the cue's local 3x3 cell neighborhood.

There is **no `proposals[N,K]` matrix**.

This is only a computational geometry instrument; it is not a claim that biology performs spatial hashing.

#### Eligibility is residue, not an experiment

A normal sparse query actually transports values across active edges. Because the receiver is additive in this toy, the transported value's signed contribution is already known from the ordinary computation.

Each used edge leaves:

```text
(source, target, signed transported contribution)
```

Eight task events later, scalar `+1/-1` outcome reward-modulates those used edges. No ablation query. No candidate perturbation query.

Weak touched edges decay below threshold and die; later geometry encounters can grow replacements.

#### Full transfer receipt

The genome evolved only at `N=48`. For every evaluation world the phenotype is erased and regrown from the genome.

Three unseen seeds (`41,42,43`):

```text
N    initial  learned-old  shift0  recovered  queryW  candidateW  edges/node  phenotypeKB
64     .714      .781       .719      .780      7.53      22.24       1.141       10.0
128    .684      .785       .715      .784      6.27      25.18       1.040       20.0
256    .704      .783       .746      .773      7.17      27.07       1.114       40.0
512    .707      .791       .736      .783      7.11      28.90       1.110       80.0
```

So in this synthetic constant-density world:

```text
inherited genome size             80 bytes, constant
represented capacity              8x growth: 64 -> 512
mean query work                   ~6--8
candidate discovery inspections   ~22 -> ~29
grown active edges/node           ~1.0--1.14
```

The phenotype memory itself is still O(N), as expected.

#### Controls

At `N=128`, three unseen seeds:

```text
condition                     learned-old   recovered
full developmental genome        .785         .784
no delayed flow credit           .758         .720
no growth                        .726         .732
static phase, matched offset     .773         .765
```

The phase result is modest. **Do not claim oscillations are essential.** Phase is presently only an optional dynamical coordinate that evolution happened to use in this toy.

#### Genome vs phenotype transfer

At unseen `N=64` worlds:

```text
erase learned graph + regrow from genome   .781
replay inherited grown graph by node id    .647
```

This is the cleanest Gate 1F-A observation:

> **What transferred in this toy was more useful as a developmental rule than as the previously acquired topology.**

#### What was actually removed

Compared with Gate 1E-A:

```text
hand-prepared candidate union     REMOVED
counterfactual ablation query     REMOVED
counterfactual candidate probe    REMOVED
useful-pair teacher               still REMOVED
```

What remains is an ordinary local substrate, sparse query, local persistent state, structural birth/death at the edge level, and delayed task consequence.

---

## Gate 1F-A guardrails

This is still a synthetic mechanism receipt.

1. **Smooth spatial task:** useful relations are deliberately correlated with geometry.
2. **One evolutionary training world:** the GA has not yet been trained across a distribution of worlds.
3. **Fixed node set:** edges are born/pruned; nodes do not divide, differentiate or die.
4. **Fixed degree / query budget:** bounded inference work is partly by construction.
5. **No hardware claim:** Python logical work only; no real cache/memory-traffic result.
6. **No strong retrieval baselines:** ANN/HNSW, sparse tables, generic routers, event-RNN/delta and MoE controls remain mandatory.
7. **No real event stream yet.**

Do not claim invention of neuroevolution, developmental encodings, NEAT/HyperNEAT-like ideas, neural developmental programs, structural plasticity, reward-modulated plasticity, evolved local learning rules, or activity-dependent growth/pruning.

The possible DifferentMachine contribution remains the measured combination and scaling frontier:

> **Can represented capacity and acquired structure grow much faster than the amount of machine that must wake, search and adapt for each event?**

---

## Immediate next experiment

Do **not** add node reproduction/death merely because the biological analogy is attractive.

The next hard step should remove the synthetic smooth-geometry favor:

> **Can the shared developmental rule create/revise useful sparse structure from a real sparse event stream, while candidate discovery, update cost and receiver query work remain bounded as stored capacity grows?**

Best substrate already in the repo family: `NeuromorphicDVSplusEMDfield`.

Why:

```text
real webcam/event coordinates -> natural geometry
DVS changes                    -> naturally sparse addressed arrivals
persistent held field          -> local temporal residue
stored identities/templates    -> capacity axis that can be scaled
```

That gives the next matched test against exhaustive template matching and ANN/HNSW rather than another graph-only toy.

---

## Files

```text
different_machine/core.py
different_machine/frontier.py
different_machine/plasticity.py
different_machine/online_plasticity.py
different_machine/flow_credit.py
different_machine/development.py

experiments/gate0_same_machine.py
experiments/gate0b_receiver_frontier.py
experiments/gate1a_coupled_frontier.py
experiments/gate1b_capacity_scaling.py
experiments/gate1c_learned_locality.py
experiments/gate1d_online_plasticity.py
experiments/gate1e_flow_credit.py
experiments/gate1f_genome_not_graph.py

tests/test_core.py
tests/test_frontier.py
tests/test_plasticity.py
tests/test_online_plasticity.py
tests/test_flow_credit.py
tests/test_development.py

docs/ARCHITECTURE.md
docs/GATE1.md
docs/GATE1B_CAPACITY_SCALING.md
docs/GATE1C_LEARNED_LOCALITY.md
docs/GATE1D_ONLINE_PLASTICITY.md
docs/GATE1E_FLOW_CREDIT.md
docs/GATE1F_GENOME_NOT_GRAPH.md
```
