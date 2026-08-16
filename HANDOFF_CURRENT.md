# DifferentMachine — CURRENT HANDOFF

**Status:** v0 executable substrate + coupled-frontier instrument committed. Gate 0/0b/1A are receipts/instruments only. Full Gate 1 controls remain unrun.

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

tests/test_core.py
tests/test_frontier.py

docs/ARCHITECTURE.md
docs/GATE1.md
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

The question is now sharp:

> **When coupling is real but not global, can a cheap receiver-conditioned active frontier preserve the needed consequence with less total executed work than ordinary conditional-compute controls?**
