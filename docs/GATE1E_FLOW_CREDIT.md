# Gate 1E-A — delayed task consequence x causal flow

Gate 1D still had one large cheat: plasticity was directly handed receiver-labelled pair co-use `(receiver, i, j)`. Even though the update itself was bounded and local, the learner was told which two endpoints belonged together.

Gate 1E-A removes that teaching event.

The experiment asks a narrower question first:

> **If a bounded local proposal neighborhood already contains useful alternatives, can the machine decide which structural relations to keep from delayed end-to-end task consequence, using only the receiver-flow change caused by a structural perturbation as eligibility?**

This is deliberately inspired by the failure ladder in `FunctionalArbors`: recent activity was too broad, exact birth-event identity was still insufficient, and the next clean eligibility candidate there was before/after flow redistribution `J_after - J_before`.

## Task

The world contains `N` addressed nodes, two receivers and receiver-specific hidden partitions. Nodes in one hidden group emit noisy evidence for one binary task sign. A query begins from one cue address and may touch only six nodes through the receiver-local degree-4 overlay. The receiver predicts from the summed evidence it actually reached.

Plasticity is **not** given:

```text
hidden group id
useful neighbor id
pair co-use event
receiver-labelled (i,j) teaching tuple
```

The environment returns only delayed scalar task outcome:

```text
+1 = receiver prediction was correct
-1 = receiver prediction was wrong
```

The outcome arrives eight later task events after the structural eligibility trace was recorded.

## Structural eligibility

Each task event measures two bounded perturbations around the addressed cue row:

```text
1. remove one currently stored edge
   -> measure receiver output change

2. temporarily install one candidate edge
   -> measure receiver output change
```

The stochastic node evidence is held identical across baseline / ablation / candidate probes, so the difference is caused by topology rather than resampled noise.

The stored trace is only:

```text
cue
existing neighbor id
existing receiver-flow delta
candidate neighbor id
candidate receiver-flow delta
receiver's own earlier prediction sign
```

When delayed task outcome arrives:

```text
credit = outcome * earlier_prediction_sign * flow_delta
```

For the binary task this is a highly informative modulatory signal, which is why the result below is only a credit-assignment receipt rather than a general reinforcement-learning result.

Only one fixed-degree row is aged/modified when credit is applied.

## The remaining proposal scaffold cheat

Gate 1B already established a hard systems condition:

> **Relevance has to be locally discoverable.**

Gate 1E-A therefore does not mix proposal discovery and credit assignment into one failure mode. Each node has a fixed 12-candidate physical proposal row. That row is constructed from a union of possible old/new receiver-0/receiver-1 neighborhoods plus distractors.

The learner sees only the candidate ids. It is not told which hidden relation produced a candidate or whether the candidate is useful for the current receiver/regime.

This is privileged world construction. It is analogous to giving a growth cone a bounded set of legal nearby structural moves and asking task consequence which move should survive.

## Verified local receipt

Two seeds (`41,42`), degree `4`, proposal degree `12`, six-node query budget, 18 old-regime epochs and 40 post-switch epochs:

```text
N      initial   learned old   immediately after switch   recovered
64       .653        .846               .616                 .874
128      .640        .874               .617                 .886
256      .637        .879               .618                 .887
```

Receiver query work remains exactly `30` logical operations in this fixed-budget instrument. Persistent graph+proposal storage in the implementation scales linearly:

```text
N=64    20 KB
N=128   40 KB
N=256   80 KB
```

Representative `N=128` post-switch learning accounting:

```text
receiver probe-query work / task event   89.0
proposal inspections / task event        12.0
bounded slot ops / task event            ~60.8
row updates / task event                  1.00
```

The `89` probe work is not hidden: Gate 1E-A spends the ordinary receiver query plus one existing-edge ablation query plus one candidate-perturbation query. It is bounded under fixed degree/query budget, but it is real learning cost.

## Controls at N=128

Same seeds and stream structure:

```text
mode                 learned old   recovered
main flow x outcome      .874         .886
outcome shuffled         .629         .602
flow only                .587         .614
reward only              .595         .656
frozen                    .625         .639
```

So neither "this structural event changed flow" nor "the task succeeded" is sufficient by itself in this instrument. The causal pairing matters.

## Narrow verdict

**[V]** The explicit `(receiver, i, j)` useful-pair teaching event can be removed in this bounded-proposal task.

**[V]** A local before/after receiver-flow trace multiplied by delayed task consequence can select receiver-relative structural relations and re-select them after a hidden regime switch without graph reset.

**[V]** Learning-time row mutation remains bounded: one addressed row per settled task event.

**[K]** "Proposal discovery is solved." False. Useful alternatives are deliberately present in the fixed 12-candidate scaffold.

**[K]** "Learning is cheap." Not established. The current causal-flow instrument pays two extra bounded receiver traversals per task event.

**[K]** "This is a general task-driven structural-plasticity result." Not established. The task is binary, synthetic and unusually informative.

## What changed conceptually

Gate 1D was:

```text
world says: (i,j) mattered for receiver r
                 -> reinforce relation
```

Gate 1E-A is:

```text
local structure changes
        |
        v
what did that change at this receiver?
        |
        v
store only that causal-flow eligibility
        |
        ... later ...
        |
        v
did the task succeed or fail?
        |
        v
stabilize / weaken the responsible local relation
```

That is the FunctionalArbors lesson transplanted into DifferentMachine: **recent is not enough; changed is not enough; eligibility should be closer to what the structural event actually changed in the downstream computation.**

## Next gate

The remaining wall is now much more specific:

> **Can candidate relations themselves arise from an ordinary real substrate / stream rather than from a hand-prepared proposal scaffold, while retaining bounded discovery and credit cost?**

That is where the repo combinations become useful: event vision, persistent memory, predictive suppression and physical/local growth can supply candidate structure without handing DifferentMachine a latent relation union.

Run:

```bash
python experiments/gate1e_flow_credit.py
```

The experiment is a mechanism receipt, not the full matched Gate 1 architecture comparison. ANN / sparse-table / delta-RNN / MoE / generic-router controls remain mandatory before any architecture claim.
