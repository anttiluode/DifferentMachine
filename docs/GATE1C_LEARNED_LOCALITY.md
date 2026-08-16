# Gate 1C — Can useful locality itself be learned?

## Question

Gate 1B showed a conditional scaling law: per-event work can remain sublinear in
represented capacity when the causal neighborhood remains bounded, and the
advantage disappears as fanout becomes global.

Gate 1C asks the next question:

> **Can experience create the locality that later makes computation cheap?**

The easy version of this question is already occupied by associative memory,
clustering, graph learning, self-organizing maps, sparse retrieval, cache/layout
optimization and many related fields. This gate is therefore a mechanism receipt,
not a novelty claim.

## Stronger version tested here

A single global notion of locality may be insufficient. The same stored item can
matter together with different items for different receivers/tasks.

So the toy contains two receiver-specific latent partitions over the *same* N
addressable nodes.

For receiver `r`, an unseen group label defines the set of items that are useful
together for that receiver. The learner never sees group labels. It sees only
receiver-labelled pairwise co-use events.

Most events for receiver `r` pair nodes from `r`'s latent group. Structured
distractors pair according to the other receiver's incompatible partition.

## Learning instrument

During acquisition, a generous sparse pair-count table records observed co-use.
After acquisition, the table is compiled to a fixed-degree local graph:

```text
node i
  -> top-d repeatedly co-used neighbors
```

The count table is then conceptually discardable; only the bounded adjacency and
edge strengths are persistent inference state.

This is *not yet* the desired online local plasticity mechanism. It asks only
whether useful locality can be inferred from experience and compiled into a
bounded substrate.

## Query

Every node is used as a cue. This prevents increased capacity from hiding as dead
padding.

Starting from cue `i`, the query may only discover nodes by traversing locally
exposed edges. It must recover at least 75% of the 8-node receiver-relevant group.

A touched node exposes exactly four stored edges. Logical query work is:

```text
node touches + local edge inspections
```

No global scan is performed by the local query.

## Conditions

### A — random bounded graph

Same degree and capacity, no learned locality.

### B — pooled learned graph

Pair counts from both receivers are merged and compiled into one universal
adjacency.

### C — receiver-conditioned learned graphs

Each receiver gets its own degree-4 relationship/locality overlay learned from
that receiver's co-use events.

This costs more persistent memory. It is not free.

### D — aligned-receiver control

Both receivers are given the *same* latent partition. If the pooled graph still
fails here, the conflict interpretation is wrong.

## Fixed parameters

```text
N                 64, 128, 256, 512
group size         8
local degree       4
receivers          2
co-use events      60 * N per receiver
signal fraction    0.80
recall target      0.75
```

With degree 4, a same-memory one-hop table can expose at most the cue + four
neighbors = 5/8 = 62.5% of the group. The 75% target therefore requires useful
multi-hop structure rather than a single direct lookup at the same stored degree.
A wider direct table remains a legitimate memory-for-compute control.

## What would count as a useful receipt?

- Receiver-conditioned learned locality reaches the recall target with query work
  that grows much more slowly than N.
- Random locality grows toward global-search cost.
- A pooled graph succeeds when receiver needs are aligned.
- A pooled graph degrades when receiver needs are incompatible.
- Persistent receiver-specific graph memory is reported and grows with N.

## Kill / demotion rules

```text
learned graph ~= random
    -> repeated co-use did not create useful locality

pooled graph works equally well under incompatible receivers
    -> receiver-relative locality is unnecessary in this toy

pooled graph fails even when receivers are aligned
    -> failure is an artifact of the learner, not task conflict

receiver-conditioned query work grows ~N
    -> learned locality did not separate capacity from retrieval work
```

## What this cannot establish

Even a clean positive receipt would not show that DifferentMachine has a new
learning algorithm. In particular:

- pair-count clustering is established;
- inference uses known receiver identity;
- acquisition uses a generous temporary count table;
- topology is compiled after learning instead of rewired continuously;
- the target is known-answer group recall rather than an end-to-end task loss;
- no delta/MoE/ANN/associative-memory retrieval baseline is yet matched;
- no hardware result follows.

The next harder question, if this receipt passes, is whether a **bounded online
plasticity rule** can learn/adapt the local substrate without a global count table,
and whether its total memory/work/reacquisition frontier beats strong sparse
retrieval and conditional-routing controls.
