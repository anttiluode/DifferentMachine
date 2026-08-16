# DifferentMachine — CURRENT HANDOFF

**Status:** v0 executable substrate + coupled-frontier instrument + capacity-scaling gate committed. Gate 0/0b/1A/1B are receipts/instruments only. Full matched Gate 1 controls remain unrun.

## One-line thesis

> **The size of the represented machine need not equal the amount of machine executed for each event.**

## Current machine

```text
C(t) persistent contact/branch substrate
z(t) local state + last-touch time
A(t) addressed / expanding active frontier
m(t) candidate/promoted outward event
k     persistent receiver<->branch relationship estimate
```

`EventMachine` performs exact lazy updates of contact, branch and receiver state.
`ClockedMachine` explicitly sweeps the entire represented state every tick.
`frontier.py` adds a coupled cable graph in which an addressed event can open a best-first causal frontier through neighboring branches under a hard branch-touch budget.

## Gate 0 — same machine, different execution

Local deterministic receipt:

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

reference NumPy timing
clocked                      ~0.043 s
lazy                         ~0.0023 s
```

Timing is not a hardware claim.

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

The exact final-readout sensitivity control is much stronger and deliberately privileged. It is a ceiling, not the mechanism.

## Gate 1A — coupled active-frontier instrument

`different_machine/frontier.py` creates a 64-branch sparse cable graph. Processing one frontier item touches one branch and may reveal child consequences. Unexpanded children are not computed.

Receiver relationship state is learned from noisy addressed interactions using simple per-branch system identification, then propagated through the known cable graph to score frontier items.

At fixed budget 48, 200 deterministic trials per receiver:

```text
rho=0.15
  r0 magnitude 2.28e-5   receiver 1.09e-5
  r1 magnitude 2.13e-5   receiver 7.08e-6

rho=0.35
  r0 magnitude 8.61e-4   receiver 3.01e-4
  r1 magnitude 5.65e-4   receiver 2.60e-4

rho=0.55
  r0 magnitude 5.37e-3   receiver 1.71e-3
  r1 magnitude 4.30e-3   receiver 1.85e-3

rho=0.75
  r0 magnitude 2.07e-2   receiver 6.15e-3
  r1 magnitude 1.57e-2   receiver 7.16e-3
```

Learned direct relationship RMSE is ~0.0107. The exact value-function policy remains an oracle ceiling.

**Do not promote this as Gate 1 positive.** This is still linear system identification + best-first conditional computation. Delta/event-RNN/MoE controls and router accounting are not in this instrument yet.

## Gate 1B — capacity vs per-event work

Question:

```text
Can represented capacity N grow much faster than required per-event work W(N, eps)
at fixed receiver error eps?
```

Every branch is used exactly twice as an originating address at every capacity, so growing `N` is not dead padding. Receiver relationship memory and acquisition samples are counted separately and grow with `N`.

Fixed setup:

```text
N                   32, 64, 128, 256, 512
rho                 0.55
propagation depth   5
target NRMSE        0.01
```

Logical event work is:

```text
branch expansions + receiver-conditioned scorer calls
```

Three topology regimes:

```text
bounded fanout      degree = 3
mesoscopic mixing   degree ~= sqrt(N)
increasing mixing   degree ~= N/8
```

Founding deterministic local receipt:

```text
bounded degree=3
N       required K   logical work
32          24           94.0
64          24           95.3
128         24           95.0
256         24           95.2
512         24           94.5
fit W ~ N^0.001

sqrt-degree
N       degree   required K   logical work
32        6          48          337.0
64        8          48          433.0
128      11          32          385.0
256      16          48          817.0
512      23          48         1153.0
fit W ~ N^0.447

N/8 mixing
N       degree   required K   logical work
32        4          24          119.7
64        8          48          433.0
128      16          48          817.0
256      32          48         1585.0
512      64          32         2081.0
fit W ~ N^1.011
```

Global-clock logical state-touch reference scales as `N^1`.

Interpretation:

> **In this toy, required per-event work follows causal fanout/frontier growth much more closely than total represented capacity.**

This gives the hypothesis a real boundary: bounded structural locality can decouple stored capacity from per-event work; increasingly global coupling destroys that decoupling because frontier discovery/scoring itself becomes linear.

This is established locality/sparse-execution territory, not a novelty result. See `docs/GATE1B_CAPACITY_SCALING.md`.

## Files

```text
different_machine/core.py
    ClockedMachine / EventMachine / ReceiverBank

different_machine/frontier.py
    sparse coupled cable graph
    exact receiver value table
    noisy relationship acquisition
    best-first budgeted frontier

experiments/gate0_same_machine.py
experiments/gate0b_receiver_frontier.py
experiments/gate1a_coupled_frontier.py
experiments/gate1b_capacity_scaling.py

tests/test_core.py
tests/test_frontier.py

docs/ARCHITECTURE.md
docs/GATE1.md
docs/GATE1B_CAPACITY_SCALING.md
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
```

The possible contribution has to survive as the **combination and measured frontier**: persistent addressed local state + expanding causal frontier + hierarchical promotion + receiver-specific relationship state, with total work tied to receiver-relevant causal change.

## Next action

Do not add another oracle toy.

Build the matched Gate 1 comparison in `docs/GATE1.md`:

```text
delta / cached change
event-RNN / EGRU-like
dynamic-k / MoE
generic learned frontier
receiver-conditioned learned frontier
```

Count scorer/router MACs and persistent relationship-state bytes.

The sharpened question is now:

> **Can a cheap receiver-conditioned active frontier preserve the bounded-locality scaling advantage after fair conditional-compute controls, when coupling is real but not globally dense?**
