# DifferentMachine — CURRENT HANDOFF

**Status:** v0 executable substrate committed; Gate 0 and Gate 0b are founding
receipts only. Gate 1 is the first experiment allowed to matter.

## One-line thesis

> **The size of the represented machine need not equal the amount of machine
> executed for each event.**

## Current machine

```text
C(t) persistent contact/branch substrate
z(t) local state + last-touch time
A(t) addressed active path on each input event
m(t) candidate/promoted outward event
k     receiver<->branch relation in the current toy
```

`EventMachine` performs exact lazy updates of contact, branch and receiver state.
`ClockedMachine` explicitly sweeps the entire represented state every tick.

## Local validation before first push

Gate 0:

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
clocked                      ~0.045 s
lazy                         ~0.0024 s
```

Timing is not a hardware claim.

Gate 0b, fixed 20% promotion budget:

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

The exact final-readout sensitivity control is much stronger and deliberately
privileged. It is a ceiling, not the mechanism.

## Files

```text
different_machine/core.py
    ClockedMachine
    EventMachine
    ReceiverBank
    machine spec / event types / budget helpers

experiments/gate0_same_machine.py
    exact clocked-vs-lazy execution receipt

experiments/gate0b_receiver_frontier.py
    fixed-budget receiver-relative foreground receipt

tests/test_core.py
    exactness/touch/budget/replay tests

docs/ARCHITECTURE.md
docs/GATE1.md
```

## Biological motivation, not implementation claim

Aizenbud et al. 2026 support two pieces of motivation:

- large, branching dendritic morphology can support electrical
  compartmentalization / semi-independent dendritic subunits;
- rich dendritic I/O is ultimately compressed into a narrow somatic/spiking
  output interface.

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
```

The possible contribution has to be the **combination and measured frontier**:
persistent addressed local state + hierarchical promotion + receiver-specific
relationship state, with work tied to receiver-relevant causal change.

## Next action

Implement Gate 1. Do not add more oracle receipts first.

The first question:

> Can a cheap learned receiver-conditioned frontier scorer beat magnitude/delta,
> event-RNN and dynamic-k controls after scorer cost is included, when true
> cross-branch coupling is nonzero?
