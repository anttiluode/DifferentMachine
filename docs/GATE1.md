# Gate 1 — learn the active frontier or kill the architecture

Gate 0 proves only exact lazy execution for a decomposable event machine.
Gate 0b proves only that a privileged receiver-conditioned budget can select a
different useful foreground.

Gate 1 is the first experiment allowed to matter.

## World

Build a streaming synthetic world with:

```text
N persistent local contacts
B branches
branch-local recurrent state
tunable cross-branch coupling rho
multiple receivers/tasks
```

Sweep independently:

```text
event density
rho = 0 -> strongly coupled
active-frontier budget
sender regime changes
receiver task changes
```

`rho=0` must not be the only regime where DifferentMachine works.

## Arms

At minimum:

```text
A dense recurrent/full-step
B delta / cached-change propagation
C event-RNN / EGRU-like
D dynamic-k / MoE router
E addressed local + magnitude/surprise promotion
F addressed local + learned generic frontier scorer
G addressed local + learned receiver-conditioned scorer
H full-information oracle ceiling
```

All arms must face the same task and total represented capacity where possible.

## Receiver-conditioned scorer

The candidate practical form is deliberately cheap:

```text
local features:
    event strength
    contact state summary
    branch state summary
    time since touch

relationship features:
    small k_(branch->receiver)

score:
    tiny bilinear/MLP function

decision:
    promote / expand frontier only if score earns budget
```

The scorer is not allowed to inspect the full global state.

## Critical perturbations

1. **Cross-branch coupling**
   Increase true nonlocal dependence. Does the frontier gracefully widen or
   collapse to full execution?

2. **Address shuffle**
   Preserve event/value statistics but send events to incorrect contacts.

3. **Private-basis/locality destruction**
   Mix the underlying causal state so physical address no longer aligns with the
   useful decomposition.

4. **Receiver switch**
   Same sender, different downstream objective. Does the learned foreground
   actually change?

5. **Familiar -> changed -> reacquired**
   Measure whether persistent relationship state reduces work when valid and
   whether communication/work rises when the sender changes.

## Accounting

Report:

```text
task quality
executed learned MACs
private-state touches
router/scorer MACs
activation bytes
persistent-state bytes
boundary traffic
promotions
reacquisition samples
CPU wall clock
GPU wall clock
```

Do not call tensor-shape accounting DRAM traffic.

## Kill rules

```text
delta/event-RNN dominates
    -> use existing event/delta machinery

MoE/dynamic-k dominates
    -> structural locality adds nothing

receiver-conditioned ~= generic
    -> relationship state adds nothing

wins only at rho=0 / almost-empty input
    -> toy decomposition

address shuffle does not hurt
    -> address-locality story is cosmetic

router tax cancels saved work
    -> no systems win

logical work falls but GPU time does not
    -> needs a different runtime/kernel before any efficiency claim
```

## Positive result

A positive Gate 1 is a reproducible Pareto improvement in a **nontrivial mixed
coupling regime** where the candidate preserves receiver task quality at lower
total executed work after router and relationship-state costs are included.

Only then reopen structural growth / pruning / WAIT-ROUTE-PROBE-GROW.
