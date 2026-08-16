# DifferentMachine — CURRENT HANDOFF

**Status:** v0 executable substrate + coupled-frontier + capacity-scaling + learned-locality receipts committed. Gate 0/0b/1A/1B/1C are still mechanism receipts/instruments, not a full matched architecture result.

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
`plasticity.py` adds the first learned-locality instrument: receiver-labelled co-use experience is compiled into bounded local adjacency.

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
receiver 0
  magnitude error            1.72e-1
  relation-score error       5.37e-2   (~3.2x lower)

receiver 1
  magnitude error            1.34e-1
  relation-score error       2.10e-2   (~6.4x lower)

relation-selected overlap across receivers
  20 / 101
  Jaccard ~0.11
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

Deterministic receipt:

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

This gate deliberately avoids the trivial question "can things be clustered?" and asks whether experience can compile a bounded local substrate that makes later work local, including a conflict where two receivers need incompatible neighborhoods over the same stored nodes.

Fixed setup:

```text
N                    64, 128, 256, 512
group size            8
local degree          4
receivers             2
co-use events         60 * N per receiver
signal fraction       0.80
query recall target   75%
```

The learner sees receiver-labelled pairwise co-use events, never latent group labels. A generous learning-time sparse pair-count table is compiled to top-4 local adjacency and is conceptually discardable afterward.

### Conflicting receiver localities

GitHub Actions result:

```text
N      pooled work   receiver-conditioned work   random work
64          137.0                 30.0               207.1
128         285.5                 30.0               399.9
256         586.2                 30.0               801.2
512        1230.2                 30.0              1603.3

fit:
pooled                W ~ N^1.054
receiver-conditioned  W ~ N^0.000
random                W ~ N^0.986
```

Each receiver-conditioned graph required exactly 6 node touches at every N. The pooled universal graph grew from ~26--29 touches at N=64 to ~240--252 at N=512.

Persistent compiled memory at N=512:

```text
one pooled degree-4 graph       32 KB
two receiver degree-4 overlays  64 KB
```

Constant query work was therefore purchased with additional relationship memory; this cost is explicit.

### Aligned-receiver control

When both receivers were assigned the same locality, one pooled graph returned to:

```text
6 touches
30 logical work
100% success
W ~ N^0.000
```

for every N.

Narrow interpretation:

> **Useful locality can be learned from repeated co-use in this toy, but locality need not be a single global property of the stored machine. It can be receiver/task-relative.**

This is associative graph learning / clustering territory, not a novelty result.

See `docs/GATE1C_LEARNED_LOCALITY.md`.

## Files

```text
different_machine/core.py
    ClockedMachine / EventMachine / ReceiverBank

different_machine/frontier.py
    sparse coupled cable graph
    exact receiver value table
    noisy relationship acquisition
    best-first budgeted frontier

different_machine/plasticity.py
    receiver-specific latent partition instrument
    pairwise co-use acquisition
    bounded top-k graph compilation
    local-only retrieval traversal

experiments/gate0_same_machine.py
experiments/gate0b_receiver_frontier.py
experiments/gate1a_coupled_frontier.py
experiments/gate1b_capacity_scaling.py
experiments/gate1c_learned_locality.py

tests/test_core.py
tests/test_frontier.py
tests/test_plasticity.py

docs/ARCHITECTURE.md
docs/GATE1.md
docs/GATE1B_CAPACITY_SCALING.md
docs/GATE1C_LEARNED_LOCALITY.md
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
```

The possible contribution has to survive as the **combination and measured frontier**: persistent addressed local state + learned bounded locality + expanding causal frontier + hierarchical promotion + receiver-specific relationship state, with total work tied to receiver-relevant causal change.

## Immediate next gate

Do **not** treat offline pair-count compilation as the desired plasticity mechanism.

Ask whether the same receiver-relative locality can emerge under a bounded online rule:

```text
O(degree) or O(degree log degree) persistent state per node/receiver
streaming edge promotion/eviction only
no global pair-count table
no reset when the world changes
reacquisition curve after regime change
same-memory flat sparse-table / ANN retrieval controls
query + update wall clock
```

If online local plasticity cannot recover the constant-work regime, the "growing cable space" story should be demoted to ordinary offline indexing.

The full matched Gate 1 conditional-compute comparison (delta / event-RNN / MoE / generic vs receiver-conditioned frontier) still remains mandatory before any architecture claim.
