# Architecture — capacity is not execution

## 1. State decomposition

DifferentMachine keeps four objects explicit:

```text
C(t)  substrate / cable space
z(t)  persistent local state on that substrate
A(t)  active causal frontier
m(t)  promoted public events
```

Optionally, each sender->receiver relation also has persistent state:

```text
k_(i->r)(t)
```

which stores what receiver `r` has learned about sender `i`.

The full represented machine may be large even when `A(t)` is small.

## 2. Contact

A contact has:

```text
address
home branch
decay
gain
weight
persistent state
last-touch time
```

On an event `(t, address, value)` only that contact is advanced from its
last-touch time to `t`, then updated nonlinearly.

Quiet contact state is **lazy**, not absent.

## 3. Branch

The contact address selects a home branch. The branch also has persistent state
and last-touch time. It is advanced only when one of its contacts produces a
local consequence.

This is the first hierarchy:

```text
many contacts -> fewer branches
```

## 4. Promotion

A touched branch creates a candidate outward event. The candidate can:

```text
die locally
promote to module/soma
promote to receiver A
promote to receiver B
...
```

A future version may use a score such as:

```text
score_i ~= local consequence
           * predicted receiver relevance(k_(i->r))
           / estimated execution cost
```

under a hard work/traffic budget.

## 5. Receiver relationship state

The same sender need not expose the same foreground to every receiver.

```text
sender i -> receiver r : k_(i->r)
sender i -> receiver q : k_(i->q)
```

`k` is not free. Any benchmark must account for its memory, updates and
reacquisition cost.

## 6. Matrix view is still valid

Nothing here rejects linear algebra.

At a fixed state, the machine can be linearized and described with a Jacobian.
The claim under test is about **execution semantics**:

```text
operator description != requirement to evaluate the full operator every event
```

## 7. Why this may fail

The idea dies if genuine coupling makes the active frontier rapidly become the
whole machine, if a router costs as much as the work it avoids, or if standard
delta/event/MoE systems dominate the same quality/compute frontier.

That is why Gate 1 sweeps cross-branch coupling instead of benchmarking only an
exactly decomposable toy.
