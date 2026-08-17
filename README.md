# DifferentMachine

> **The size of the represented machine need not equal the amount of machine executed for each event.**

`DifferentMachine` is an executable research program around one systems question:

> **Can a large learned machine keep most of its capacity as persistent local possibility, while the work performed for each event follows only the receiver-relevant causal frontier?**

It is not a claim that event-driven computing, sparse retrieval, structural plasticity, neuroevolution, spiking, MoE, or developmental neural systems are new.

The repo is deliberately built as a sequence of killable gates rather than one large architecture claim.

## Current picture

```text
small shared developmental rule / genome
        |
        v
persistent substrate + local state
        |
addressed event / local encounter
        |
        v
small active causal frontier
        |
        v
receiver-specific consequence
        |
        +-- ordinary participation leaves local eligibility
        +-- delayed task consequence stabilizes / weakens structure
        `-- weak structure prunes; local encounters can regrow it
```

The current conceptual state is:

```text
C(t)  persistent substrate / acquired structure
z(t)  persistent local state
A(t)  active causal frontier
m(t)  promoted receiver consequence
e(t)  local eligibility left by actual signal transport
theta small shared developmental rule
```

The newest distinction is:

> **genome != acquired graph phenotype**

The graph can be destroyed while the small inherited rule remains able to grow another one.

## Gate ladder

| Gate | Question | Narrow receipt |
|---|---|---|
| 0 | Must all persistent state be explicitly updated every tick? | No: exact lazy execution matched the clocked machine while touching dramatically less quiet state. |
| 0b | Can the active foreground depend on the receiver? | In a toy hard-budget readout, receiver relation beats pure magnitude. |
| 1A | Can a causal frontier expand through coupled structure? | Yes as an instrument; still missing strong matched baselines. |
| 1B | Can capacity grow faster than work/event? | With bounded local degree, `W ~ N^0.001`; when fanout becomes global the advantage dies. |
| 1C | Can useful locality itself be learned? | Receiver-specific learned overlays stay flat-work while pooled/random graphs approach linear work. |
| 1D | Can locality reorganize online with bounded updates? | Yes in a synthetic co-use world; after hidden change work spikes toward global search, then returns to flat scaling after relearning. |
| 1E-A | Can delayed task consequence replace the useful-pair teacher? | Yes in a toy when causal flow is measured explicitly; but candidate lists and counterfactual probe cost remain. |
| 1F-A | Can a small shared rule grow the graph instead of inheriting it? | Yes in the current synthetic spatial task: geometry creates candidates, normal flow leaves eligibility, edges grow/prune, and an 80-byte evolved rule transfers across unseen worlds and larger `N`. |

See [`HANDOFF_CURRENT.md`](HANDOFF_CURRENT.md) for the live research state and caveats.

## Gate 1B: the scaling constraint

The central scaling result that constrains everything after it is:

```text
bounded degree=3       W ~ N^0.001
sqrt(N) degree          W ~ N^0.447
N/8 degree              W ~ N^1.011
global clock reference  W ~ N^1.000
```

So the design rule is not merely "be sparse":

> **Relevance must be locally discoverable.**

If discovering relevance itself requires a global scan, the machine has only moved the cost.

## Gate 1D: experience amortizes future work into structure

With fixed-degree local online plasticity:

```text
relationship valid      -> local / approximately constant work
relationship invalid    -> near-global search work
relationship relearned  -> local / approximately constant work again
```

This suggests a useful systems interpretation:

> **Repeated interaction can amortize future computation into persistent local structure.**

## Gate 1E-A: task consequence without a useful-pair teacher

Gate 1E-A removed the explicit `(receiver, i, j)` co-use teacher and used delayed task outcome to select topology.

It passed a synthetic causal-control screen, but it still paid for:

```text
normal receiver query
+ existing-edge ablation query
+ candidate perturbation query
```

and still received a hand-prepared bounded candidate scaffold.

That was intentionally not the endpoint.

## Gate 1F-A: genome, not graph

Gate 1F-A removes those two conveniences from the current instrument.

### Candidate generation

Nodes live in a constant-density 2-D substrate. Candidate relations are produced by a fixed-radius spatial hash around the addressed node.

There is no hand-supplied `proposals[N,K]` matrix.

### Eligibility

When an ordinary sparse query transports a value across an edge, that transport leaves a small local eligibility residue. Delayed scalar success/failure later modulates only edges that actually participated.

There is no ablation or candidate counterfactual query in Gate 1F-A.

### Inherited object

A deterministic mutation-only genetic algorithm evolves ten shared float64 values:

```text
10 genes = 80 bytes
training size N = 48
```

Those genes control local growth/pruning dynamics; they contain no node ids or per-edge weights.

The canonical search is reproducible:

```bash
python experiments/gate1f_genome_not_graph.py --evolve
```

### Transfer receipt

The phenotype is erased before every unseen world. The same genome is then regrown at larger capacities:

```text
N    initial  learned-old  shift0  recovered  queryW  candidateW  edges/node
64     .714      .781       .719      .780      7.53      22.24       1.141
128    .684      .785       .715      .784      6.27      25.18       1.040
256    .704      .783       .746      .773      7.17      27.07       1.114
512    .707      .791       .736      .783      7.11      28.90       1.110
```

The genome was evolved only at `N=48`. The represented capacity sweep above reaches more than 10x that evolutionary training size while the inherited rule remains 80 bytes.

At `N=64`:

```text
erase graph + regrow from genome   .781
replay inherited grown graph       .647
```

So in this toy, the transferable object is more useful as a **developmental rule** than as the previously learned adjacency itself.

Detailed write-up: [`docs/GATE1F_GENOME_NOT_GRAPH.md`](docs/GATE1F_GENOME_NOT_GRAPH.md).

## Biological motivation, not implementation claim

The biology is useful as a source of decomposition questions:

```text
rich local persistent dynamics
narrow outward consequences
activity-dependent structural change
multiple timescales
growth / stabilization / pruning
```

`DifferentMachine` does **not** claim to simulate a neuron, dendrite, gene regulatory network, or developing brain.

The current 1F substrate has edge birth and edge death only. Nodes do not yet divide, differentiate, reproduce, or die.

## What still has to be beaten

A real architecture result must compare against strong ordinary alternatives, including:

- ANN/HNSW or sparse-table retrieval;
- learned routers / conditional computation;
- delta / temporal-sparsity inference;
- event-RNN / activity-sparse recurrent execution;
- MoE-style routing;
- dense recurrent / SSM baselines where appropriate.

And it must pay for:

- candidate discovery;
- router/scorer work;
- persistent state bytes;
- update cost;
- synchronization;
- reacquisition after change;
- actual CPU/GPU wall clock and memory traffic.

## Current largest cheats

Gate 1F-A is still favorable to itself:

1. the synthetic task is spatially smooth, so proximity is genuinely informative;
2. the GA evolved on one small training world rather than a distribution;
3. degree and query budget are fixed;
4. the node population is fixed;
5. there is no real-stream or hardware result yet.

The next useful test is therefore **not more biological machinery**. It is a real sparse event substrate where geometry is supplied by the world rather than designed for the task.

The strongest existing candidate in the repo family is `NeuromorphicDVSplusEMDfield`: real image/event coordinates can generate candidacy naturally, while stored identity/template capacity supplies a practical scaling axis.

## Quick start

```bash
python -m pip install numpy pytest
pytest -q

python experiments/gate0_same_machine.py
python experiments/gate1b_capacity_scaling.py
python experiments/gate1d_online_plasticity.py
python experiments/gate1f_genome_not_graph.py

# heavier transfer sweep
python experiments/gate1f_genome_not_graph.py --full

# reproduce the tiny genetic search
python experiments/gate1f_genome_not_graph.py --evolve
```

Python 3.11 and 3.13 are exercised in GitHub Actions.

## Status

**Research prototype / mechanism receipts. No novelty or production-performance claim.**

The live handoff is [`HANDOFF_CURRENT.md`](HANDOFF_CURRENT.md).
