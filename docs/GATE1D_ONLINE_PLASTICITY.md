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

## Pass pattern

A useful mechanism receipt requires:

- low steady-state query work with `W ~ N^alpha`, alpha near 0;
- immediate post-change work growing toward global-search scaling;
- recovery to near-flat query work after a roughly constant number of new events per node;
- O(degree) update work independent of N;
- explicit linear persistent-memory cost;
- no-forgetting and recency controls fail to recover the same query frontier.

## What this would mean

If the gate passes, the narrow statement is:

> **A large persistent substrate can be reorganized by bounded endpoint-local updates so that future receiver-relative retrieval becomes local again after experience changes.**

That would make the resource separation concrete:

```text
stored capacity             ~ N
relationship memory         ~ N
online update work/event    ~ O(degree)
steady query work/event     potentially ~ N^0
reacquisition data          ~ constant events/node, total ~ N
```

## What it does not mean

Even a clean pass does not establish a new learning algorithm or a superior AI architecture.

This gate remains a synthetic associative world. The learner is given receiver identity and pair co-use events. There is no end-to-end task loss, no learned representation, no ANN/sparse-retrieval baseline, no delta/event-RNN/MoE comparison, and no hardware-memory-traffic result.

The next serious gate must move from known-answer group recall to a real task where useful structure must emerge from task loss while all update/query costs are matched against strong conditional-compute and sparse-retrieval controls.
