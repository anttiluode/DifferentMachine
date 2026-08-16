# DifferentMachine

> **The size of the machine is not necessarily the amount of machine that must run.**

`DifferentMachine` is an executable research sketch for a different AI execution
primitive:

```text
persistent substrate / cable space
        +
persistent local state
        +
addressed events
        |
        v
small active causal frontier
        |
        v
hierarchical promotion
        |
        v
receiver-specific public consequence
```

This is **not** a claim that matrix mathematics is wrong, that the brain is
mostly inactive, or that event-driven neural computation is new. Spiking
networks, AER, event-RNNs, delta inference, conditional computation and Mixture
of Experts already occupy large parts of this territory.

The question is narrower:

> **Can a large learned machine keep most of its capacity as persistent local
> possibility, while executed work scales with the receiver-relevant causal
> frontier rather than total represented capacity?**

## Why this repo exists

Recent work in `GeometricNeuronV23`, `Y`, `Z`, `PivotPoint`, `WidePresent` and
`MoireMandelbrot` converged on four distinctions:

```text
C(t)  persistent substrate: what computation can happen
z(t)  local persistent state: unfinished history in the substrate
A(t)  active frontier: what is being strongly computed now
m(t)  promoted event: what consequence crosses a boundary
```

A conventional dense step often makes these look like one object:

```text
h_next = phi(W h)
```

`DifferentMachine` keeps them separate.

The first implementation compiles contact -> branch routing into persistent
structure. An input event touches one addressed contact, updates that contact's
private state, updates its home branch, and only then creates a candidate public
event. Quiet contacts and branches keep state without being swept every clock
tick.

## Gate 0: same machine, different execution

`experiments/gate0_same_machine.py` runs the **same mathematical machine** two
ways:

1. `ClockedMachine`: explicitly decays every contact, branch and receiver every
   tick.
2. `EventMachine`: advances only the addressed contact/branch and lazily catches
   that state up when it is next touched.

Local deterministic receipt:

```text
contacts                       : 4096
branches                       : 64
ticks                          : 20,000
input events                   : 382

max candidate mismatch         : 1.11e-16
max final receiver mismatch    : 4.16e-17

clocked private-state touches  : 83,200,000
addressed/lazy touches         : 764
logical touch ratio            : 108,900x
```

The local NumPy reference was also faster, but **that wall-clock ratio is not a
hardware result**. Dense vectorized kernels and event runtimes have very
different implementation economics. Gate 0 proves only the execution identity:
quiet private state can persist without being explicitly recomputed on every
global tick.

Run:

```bash
python experiments/gate0_same_machine.py
```

## Gate 0b: the active frontier can be receiver-relative

`experiments/gate0b_receiver_frontier.py` gives the same sender a hard promotion
budget and compares:

```text
magnitude score
    |candidate|

relationship score
    |candidate * receiver_branch_relation|

privileged oracle ceiling
    exact contribution to a chosen final receiver readout
```

At a 20% promotion budget in the deterministic receipt, the simple persistent
receiver/branch relationship score reduced final-readout error versus pure
magnitude selection by about `3.2x` for receiver 0 and `6.4x` for receiver 1.
The two receivers selected almost disjoint foregrounds (Jaccard ~0.11) from the
same sender trace.

This is **not a result** yet. The relationship weights are available to the toy
machine, and the final-readout oracle is deliberately privileged. It only earns
the next experiment: learn/adapt the relation cheaply and compare it against
strong conditional-compute controls.

Run:

```bash
python experiments/gate0b_receiver_frontier.py
```

## The primitive

```python
event = InputEvent(t=..., address=..., value=...)

machine = EventMachine(spec)
candidate = machine.process_event(event)
```

Internally:

```text
addressed contact
    persistent local state
        |
        v
home branch
    persistent branch state
        |
        v
candidate outward event
        |
        +-- die locally
        `-- promote to receiver(s)
```

Routing is already partly compiled into structure: an event does not run a
global router to discover its first destination. The contact address selects the
local state and home branch.

## Why the dendrite analogy is useful

Aizenbud et al. (PNAS, 2026) report that larger dendritic surface/extent and
branching are associated with greater modeled single-neuron functional
complexity, and discuss electrical compartmentalization that allows dendritic
regions to act as semi-independent computational subunits. Their complexity
assay ultimately evaluates somatic spike prediction.

That motivates—but does not prove—the systems decomposition used here:

```text
rich private local processing
        ->
narrow outward event interface
```

Reference:

> Aizenbud et al. (2026), *Dendritic morphology and synaptic nonlinearities
> enhance functional complexity in human cortical neurons*, PNAS 123(28),
> e2533168123. DOI: 10.1073/pnas.2533168123.

## What must be beaten

A real `DifferentMachine` result has to beat or explain itself relative to:

- dense recurrent / SSM execution,
- delta / temporal-sparsity inference,
- activity-sparse event-RNN / EGRU-like execution,
- dynamic-k / Mixture-of-Experts routing,
- ordinary learned magnitude/surprise gating,
- task-oriented bottlenecks / learned communication.

And it must pay for:

- router/scorer MACs,
- persistent local/relationship-state bytes,
- state-update cost,
- synchronization,
- reacquisition after sender/task changes,
- actual CPU/GPU wall clock.

See [`docs/GATE1.md`](docs/GATE1.md).

## Current design sentence

> **Capacity lives in persistent structure and local state. Computation wakes
> where an event lands, expands only along a budgeted causal frontier, and
> crosses a boundary only when the receiver needs the consequence.**

That is the machine we are trying to build.

## Quick start

```bash
python -m pip install numpy pytest
pytest -q
python experiments/gate0_same_machine.py
python experiments/gate0b_receiver_frontier.py
```

Python 3.11 and 3.13 are exercised in GitHub Actions.

## Status

**v0 / founding receipts. No novelty claim.**

The next code allowed to matter is Gate 1: a learnable receiver-conditioned
frontier scorer under genuine cross-branch coupling, compared at matched quality
and matched compute against the controls above.
