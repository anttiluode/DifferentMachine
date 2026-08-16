# Gate 1D — Can cable space grow online, locally, with bounded plasticity?

## Question

Gate 1C showed that receiver-relative locality can be inferred from repeated co-use and compiled into a bounded graph, but its learner used a generous temporary pair-count table and an offline compile step.

Gate 1D removes that escape hatch:

> **Can each node maintain only a tiny fixed local neighborhood, update it online from experience, and still reorganize fast enough that later computation stays local?**

## Online rule

Each receiver has a degree-4 local overlay. Node `i` stores only:

```text
4 neighbor ids
4 association strengths
```

When receiver-labelled co-use event `(i, j)` arrives:

```text
age row i locally
reinforce i->j if present
otherwise use a free slot or evict weakest edge

age row j locally
reinforce j->i if present
otherwise use a free slot or evict weakest edge
```

Only rows `i` and `j` are touched. No global pair table exists and no global aging sweep occurs.

The current rule uses multiplicative local decay (`0.97`) when a row is touched. This is a simple bounded associative/plasticity heuristic, not a novelty claim.

## Fixed resources

```text
nodes N                    64, 128, 256, 512
group size                 8
receivers                  2
local degree               4
signal probability         0.80
old-regime acquisition     60 events/node/receiver
recall target              75%
seeds                       41, 42, 43
```

Persistent graph memory is exactly linear in N and fixed per node/receiver. With int64 neighbor ids and float64 strengths:

```text
2 receivers * N * 4 slots * 16 bytes/slot
```

So memory is 8 KB at N=64 and 64 KB at N=512.

Update work is bounded by degree. One pair event updates exactly two rows. The reference implementation counts row-slot operations; with degree 4 the theoretical maximum is fixed and independent of N.

## Hidden regime change

After the old relation is learned, the receiver-specific latent partitions are replaced by new independent partitions. The graph is **not reset**.

We measure query work for the new regime immediately and after:

```text
0, 8, 16, 24, 32, 48, 64
```

new co-use events per node/receiver.

This asks whether a wrong relationship state produces a transient search/computation explosion and whether bounded local plasticity can re-form a cheap substrate.

## Controls

### No-forgetting control

Same bounded graph, same memory, same update pattern, but `decay=1.0`. Old strong associations never locally lose weight.

### Recency-only control

Same storage shape, but each row stores only the most recently observed distinct neighbors. It adapts quickly but does not integrate repeated evidence against structured distractors.

## GitHub Actions receipt

Averaged over the three deterministic seeds:

```text
N    steady   shift0    e8      e16     e24     e32     e48     e64    KB   slotOps/pair
64     30.0    196.7    202.5    154.9     60.2     32.1     30.6     30.0     8       25.65
128    30.0    379.1    382.6    309.4     94.6     35.5     30.5     30.1    16       25.75
256    30.0    739.4    778.3    593.3    142.9     42.8     31.4     30.1    32       25.83
512    30.0   1497.7   1556.4   1185.1    211.9     38.2     30.6     30.4    64       25.88
```

Scaling fits:

```text
steady learned relation             W ~ N^0.000
immediately after hidden change     W ~ N^0.975
after 48 new events/node            W ~ N^0.004
```

The machine therefore moves through three qualitatively different compute regimes without changing its stored capacity:

```text
known relationship       -> local / nearly constant query work
relationship invalid     -> near-global search-like work
relationship reacquired  -> local / nearly constant query work again
```

At the same time, online update work remains fixed: about 25.6--25.9 counted slot operations per pair over the full runs, with exactly two endpoint-row updates per pair. Persistent memory grows from 8 KB to 64 KB as N grows 8x.

### Controls after regime change

```text
N    no-forgetting@96/node   recency-only@48/node   online-decay@48/node
64            202.0                  123.2                   30.6
128           390.1                  187.3                   30.5
256           787.4                  381.8                   31.4
512          1587.0                  779.3                   30.6
```

Control fits:

```text
no forgetting   W ~ N^0.993
recency only    W ~ N^0.901
```

So the recovery is not explained simply by keeping recent partners, and local forgetting is necessary for this particular bounded-slot rule to relinquish the old relationship.

## Narrow interpretation

Gate 1D passes as a **mechanism receipt**:

> **A large persistent substrate can be reorganized by bounded endpoint-local updates so that future receiver-relative retrieval becomes local again after experience changes.**

In this toy, roughly 32--48 new co-use events per node are enough to restore the constant-work regime across N=64..512. Total reacquisition data still scales with N; what stays bounded is local update work and the number of experiences needed per node.

This makes the resource separation concrete:

```text
stored capacity             ~ N
relationship memory         ~ N
online update work/event    ~ O(degree)
steady query work/event     ~ N^0 in this toy
wrong-model query work      ~ N^1 in this toy
reacquisition data          ~ constant events/node, total ~ N
```

One useful systems interpretation is that repeated interaction can **amortize future computation into persistent local structure**. When the world changes, that amortization becomes wrong and query cost spikes until plasticity rebuilds the relationship.

## What it does not mean

This does not establish a new learning algorithm or a superior AI architecture.

The gate remains a synthetic associative world. The learner is given receiver identity and pair co-use events. There is no end-to-end task loss, no learned representation, no ANN/sparse-retrieval baseline, no delta/event-RNN/MoE comparison, and no hardware-memory-traffic result.

There is also strong prior art on local/activity-dependent structural plasticity and online rewiring. DifferentMachine must not claim structural plasticity itself as new.

The next serious gate must move from known-answer group recall to a task where useful structure has to emerge from task loss while all update/query costs are matched against strong conditional-compute and sparse-retrieval controls.
