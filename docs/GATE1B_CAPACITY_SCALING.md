# Gate 1B — Capacity vs per-event executed work

## Question

> **Can represented machine capacity grow much faster than the amount of machine
> that must execute for one event?**

Write the required event work at fixed receiver error tolerance `eps` as

```text
W(N, eps) ~ N^alpha
```

where `N` is represented branch capacity.

A dense/global clocked execution has the reference behavior `alpha ~= 1`.
DifferentMachine becomes interesting only if a nontrivial coupled regime permits
`alpha < 1` without making the added capacity dead or unused.

## Guard against fake capacity

Every branch is used exactly twice as an originating event address in each
capacity condition. Receiver weights remain nonzero across the substrate.
Increasing `N` therefore increases genuinely addressable learned state rather
than appending never-used padding.

Persistent relationship state is *not free*: its bytes and acquisition samples
are reported and scale with `N`.

## Fixed task

```text
capacities N       32, 64, 128, 256, 512
receivers          2
rho                0.55
propagation depth  5
target NRMSE       0.01
relation samples   24 * N per receiver
```

At every `N`, the experiment finds the smallest active-frontier branch-touch
budget from a fixed budget ladder that meets the same 1% normalized receiver
error target.

Logical event work is counted as:

```text
branch expansions + receiver-conditioned scorer calls
```

This deliberately still omits heap/runtime overhead and is **not** a wall-clock
or FLOP result.

## Three topology scaling regimes

### 1. Bounded causal degree

```text
degree = 3
```

Total substrate capacity grows while each local site retains bounded fanout.
This is the regime in which a finite causal frontier could remain nearly
independent of total capacity.

### 2. Mesoscopic mixing

```text
degree ~= sqrt(N)
```

The number of possible immediate consequences grows sublinearly with capacity.
Even if only a few consequences are eventually expanded, the current reference
implementation must score newly exposed neighbors, so routing/frontier discovery
itself gets more expensive.

### 3. Increasingly global mixing

```text
degree ~= N / 8
```

Fanout grows proportionally with substrate size. If the execution advantage is
really caused by causal locality, the scaling should approach the dense/global
reference here.

## Founding local receipt

Deterministic prototype results at 1% target NRMSE:

```text
bounded degree=3
N       required K   scorer calls   logical work
32          24           70.0          94.0
64          24           71.3          95.3
128         24           71.0          95.0
256         24           71.2          95.2
512         24           70.5          94.5

fit: W ~ N^0.001
```

```text
sqrt-degree
N       degree   required K   scorer calls   logical work
32        6          48          289.0         337.0
64        8          48          385.0         433.0
128      11          32          353.0         385.0
256      16          48          769.0         817.0
512      23          48         1105.0        1153.0

fit: W ~ N^0.447
```

```text
N/8 mixing
N       degree   required K   scorer calls   logical work
32        4          24           95.7         119.7
64        8          48          385.0         433.0
128      16          48          769.0         817.0
256      32          48         1537.0        1585.0
512      64          32         2049.0        2081.0

fit: W ~ N^1.011
```

The global-clock logical state-touch reference scales exactly as `N^1`.

## What this earns

The receipt supports a narrow systems statement:

> **In this coupled toy, the scaling of required per-event work follows the
> growth of the causal frontier/fanout much more closely than it follows total
> represented capacity.**

That is the resource separation DifferentMachine is trying to exploit:

```text
stored capacity / relationship memory     can grow with N
per-event executed computation            may grow more slowly
```

It also gives the idea a clean failure boundary. As coupling becomes increasingly
global, merely discovering/scoring possible consequences becomes an `O(N)` cost
and the apparent advantage collapses.

## What this does NOT earn

This is not novel by itself. Bounded-degree local computation, sparse graph
execution, event-driven systems, conditional computation and database/embedding
lookup already demonstrate that stored capacity need not imply full execution.

Gate 1B also does not compare real kernels or strong learned controls.

The next meaningful result still requires the matched Gate 1 benchmark:

```text
delta / cached-change baseline
event-RNN / EGRU-like baseline
dynamic-k / MoE baseline
generic learned frontier scorer
receiver-conditioned learned frontier scorer
```

with scorer/router MACs, bytes, reacquisition, and actual CPU/GPU runtime counted.

## Design consequence

The strongest current hypothesis is no longer merely "sparsity".

It is:

> **DifferentMachine can scale only if useful knowledge can be stored in a large
> persistent substrate while causal access remains sufficiently local that the
> relevant frontier can be discovered without scanning the substrate.**

That locality/structure requirement is now part of the hypothesis, not an
implementation detail.
