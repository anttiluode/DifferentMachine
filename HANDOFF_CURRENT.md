# DifferentMachine — CURRENT HANDOFF

**Status:** v0 executable substrate + coupled-frontier + capacity-scaling + learned-locality + bounded-online-plasticity receipts committed. Gate 0/0b/1A/1B/1C/1D are still mechanism receipts/instruments, not a full matched architecture result.

## One-line thesis

> **The size of the represented machine need not equal the amount of machine executed for each event.**

## Current machine

```text
C(t) persistent contact/branch substrate
z(t) local state + last-touch time
A(t) addressed / expanding active frontier
m(t) candidate/promoted outward event
k     persistent receiver<->branch relationship state / locality overlay
```

`EventMachine` performs exact lazy updates of contact, branch and receiver state.
`ClockedMachine` explicitly sweeps the entire represented state every tick.
`frontier.py` adds a coupled cable graph in which an addressed event can open a best-first causal frontier through neighboring branches under a hard branch-touch budget.
`plasticity.py` adds offline learned-locality instruments.
`online_plasticity.py` adds a fixed-degree receiver-local graph that changes only at the two endpoints of each co-use event; local forgetting, reinforcement, insertion and eviction are all O(degree) and require no global pair table or global aging sweep.

## Gate 0 — same machine, different execution

```text
4096 contacts
64 branches
20,000 ticks
382 input events
max candidate mismatch       1.11e-16
max final receiver mismatch  4.16e-17
clocked private touches      83,200,000
lazy private touches         764
logical touch ratio          108,900x
```

Timing is a reference implementation result only, not a hardware claim.

## Gate 0b — same sender, receiver-relative promotion

Fixed 20% promotion budget:

```text
receiver 0: magnitude error 1.72e-1  relation-score error 5.37e-2 (~3.2x lower)
receiver 1: magnitude error 1.34e-1  relation-score error 2.10e-2 (~6.4x lower)
selected overlap 20/101, Jaccard ~0.11
```

The exact final-readout sensitivity control is a privileged ceiling.

## Gate 1A — coupled active-frontier instrument

A 64-branch sparse cable graph opens child consequences only when a frontier item is expanded. Learned receiver relation beats magnitude/generic scoring in the deterministic linear instrument from weak through strong nonzero coupling (`rho=0.15..0.75`).

**Do not promote this as Gate 1 positive.** It is still system identification + best-first conditional computation and has not beaten delta/event-RNN/MoE controls.

## Gate 1B — capacity vs per-event work

Question:

```text
Can represented capacity N grow much faster than required per-event work W(N, eps)
at fixed receiver error eps?
```

Every branch is exercised as an originating address. Relationship memory/acquisition grow with `N` and are accounted separately.

```text
bounded degree=3       W ~ N^0.001
sqrt(N) degree          W ~ N^0.447
N/8 degree              W ~ N^1.011
global clock reference  W ~ N^1.000
```

At bounded degree, required work stayed ~94--95 logical operations while capacity grew 16x from 32 to 512 branches. As fanout became global, frontier discovery itself became linear and the advantage vanished.

Current design constraint:

> **Relevance must be locally discoverable.**

See `docs/GATE1B_CAPACITY_SCALING.md`.

## Gate 1C — can useful locality itself be learned?

Receiver-labelled co-use is compiled offline to degree-4 local graphs. Two receivers deliberately need incompatible neighborhoods over the same nodes.

```text
N      pooled work   receiver-conditioned work   random work
64          137.0                 30.0               207.1
128         285.5                 30.0               399.9
256         586.2                 30.0               801.2
512        1230.2                 30.0              1603.3

pooled                W ~ N^1.054
receiver-conditioned  W ~ N^0.000
random                W ~ N^0.986
```

When receivers are aligned, one pooled graph returns to exactly 6 touches / 30 work / `W ~ N^0`.

Narrow interpretation:

> **Useful locality can be learned from repeated co-use in this toy, but locality need not be a single global property of the stored machine. It can be receiver/task-relative.**

This is associative graph learning / clustering territory, not a novelty result.

See `docs/GATE1C_LEARNED_LOCALITY.md`.

## Gate 1D — can cable space grow online, locally, with bounded plasticity?

Gate 1D removes Gate 1C's global-ish learning-time count table and offline compile step.

Each receiver/node stores only 4 neighbor ids and 4 strengths. Pair event `(i,j)` touches only rows `i` and `j`:

```text
locally age the four slots
reinforce existing relation
or insert / evict the weakest local relation
```

No global pair table. No global decay sweep. Degree remains fixed.

Fixed setup:

```text
N                       64, 128, 256, 512
group size               8
receivers                2
degree                   4
signal fraction           0.80
old-regime experience     60 events/node/receiver
local decay               0.97
recall target             75%
seeds                     41,42,43
```

After learning, both receiver-specific hidden partitions are replaced by independent new partitions. Graphs are not reset.

### GitHub Actions result

```text
N    steady   shift0    e8      e16     e24     e32     e48     e64    KB   slotOps/pair
64     30.0    196.7    202.5    154.9     60.2     32.1     30.6     30.0     8       25.65
128    30.0    379.1    382.6    309.4     94.6     35.5     30.5     30.1    16       25.75
256    30.0    739.4    778.3    593.3    142.9     42.8     31.4     30.1    32       25.83
512    30.0   1497.7   1556.4   1185.1    211.9     38.2     30.6     30.4    64       25.88
```

Scaling:

```text
steady learned relationship            W ~ N^0.000
immediately after hidden change        W ~ N^0.975
after 48 new events/node               W ~ N^0.004
```

So without changing capacity, the machine passes through:

```text
relationship valid      -> local / constant work
relationship invalid    -> near-global search work
relationship relearned  -> local / constant work again
```

Online update work remains bounded at ~25.6--25.9 counted slot operations per pair and exactly two endpoint-row updates per pair. Persistent relationship memory scales linearly: 8 KB at N=64 to 64 KB at N=512.

### Controls

```text
N    no-forgetting@96/node   recency-only@48/node   online-decay@48/node
64            202.0                  123.2                   30.6
128           390.1                  187.3                   30.5
256           787.4                  381.8                   31.4
512          1587.0                  779.3                   30.6

no forgetting   W ~ N^0.993
recency only    W ~ N^0.901
```

Thus simple recency does not explain recovery in this structured-distractor world, and forgetting is necessary for the current bounded-slot rule to relinquish the old relation.

**Narrow Gate 1D receipt:**

> **A large persistent substrate can be reorganized by bounded endpoint-local updates so future receiver-relative retrieval becomes local again after experience changes.**

Roughly 32--48 new events per node reacquire the flat-work regime across N=64..512. Total reacquisition data is still O(N); the constant quantity is experience per node and update work per event.

A useful systems interpretation is:

> **Repeated interaction can amortize future computation into persistent local structure. When the world changes, that amortization becomes wrong, computation/search spikes, and local plasticity rebuilds the cheap route.**

This is still established associative/structural-plasticity territory, not a novelty result.

See `docs/GATE1D_ONLINE_PLASTICITY.md`.

## Files

```text
different_machine/core.py
different_machine/frontier.py
different_machine/plasticity.py
different_machine/online_plasticity.py

experiments/gate0_same_machine.py
experiments/gate0b_receiver_frontier.py
experiments/gate1a_coupled_frontier.py
experiments/gate1b_capacity_scaling.py
experiments/gate1c_learned_locality.py
experiments/gate1d_online_plasticity.py

tests/test_core.py
tests/test_frontier.py
tests/test_plasticity.py
tests/test_online_plasticity.py

docs/ARCHITECTURE.md
docs/GATE1.md
docs/GATE1B_CAPACITY_SCALING.md
docs/GATE1C_LEARNED_LOCALITY.md
docs/GATE1D_ONLINE_PLASTICITY.md
```

## Biological motivation, not implementation claim

Aizenbud et al. 2026 motivate separating rich dendritic processing from narrow outward spike communication: morphology can support electrical compartmentalization / semi-independent dendritic subunits, while their functional-complexity assay ultimately evaluates somatic spike prediction.

DifferentMachine does **not** claim to simulate a biological neuron.

## Prior-art guardrail

Do not claim invention of:

```text
event-driven neural computation
AER
spiking
delta inference
conditional computation
MoE
activity-sparse RNNs
predictive/event-triggered communication
best-first search / value-based prioritization
bounded-degree local computation
associative graph learning / clustering
sparse retrieval / indexing
local / activity-dependent structural plasticity
online graph rewiring
```

The possible contribution has to survive as the **combination and measured frontier**: persistent addressed local state + learned/adaptive bounded locality + expanding causal frontier + hierarchical promotion + receiver-specific relationship state, with total work tied to receiver-relevant causal change.

## Immediate next gate

Gate 1D still receives receiver-labelled pair co-use as an explicit teaching signal. That is now the biggest cheat.

The next experiment should ask:

> **Can useful local topology emerge from end-to-end task consequences rather than being told which pair co-occurred?**

Requirements:

```text
streaming task, not known group recall
no latent pair label supplied to plasticity
local eligibility / credit state only
receiver/task switch without reset
same fixed degree and bounded update memory
ANN / sparse-table retrieval controls
delta / event-RNN / MoE / generic-router controls
actual query + update wall clock and memory traffic
```

If task-driven local credit cannot recover the bounded-locality advantage, DifferentMachine remains a useful systems decomposition / learned index rather than a new AI execution architecture.

The full matched Gate 1 conditional-compute comparison remains mandatory before any architecture claim.
