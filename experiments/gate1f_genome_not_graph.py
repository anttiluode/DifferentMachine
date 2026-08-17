from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import math

import numpy as np

from different_machine.development import (
    DevelopmentalGenome,
    DevelopmentalGraph,
    EdgeEligibility,
    SpatialHash,
)

N_RECEIVERS = 2
DEGREE = 4
QUERY_BUDGET = 6
MU = 0.55
REWARD_DELAY = 8
DENSITY = 0.75
RADIUS = 2.15

# Canonical deterministic search receipt: population 14, generations 9,
# evolution seed 7, training N=48, training world seed 101.
CANONICAL_RAW = np.asarray(
    [
        -0.31887550,
        +0.00685164,
        +0.69949635,
        -0.28542975,
        -1.14091092,
        -1.51895493,
        -0.69230545,
        +0.44838466,
        +1.39977894,
        +0.59710920,
    ],
    dtype=np.float64,
)


@dataclass
class World:
    positions: np.ndarray
    old_labels: list[np.ndarray]
    new_labels: list[np.ndarray]
    spatial: SpatialHash


def make_world(n_nodes: int, seed: int) -> World:
    """Constant-density 2-D substrate with receiver-specific smooth tasks."""
    rng = np.random.default_rng(seed)
    side = math.sqrt(n_nodes / DENSITY)
    positions = rng.uniform(0.0, side, size=(n_nodes, 2))

    regimes: list[list[np.ndarray]] = []
    for _ in range(2):
        receiver_labels: list[np.ndarray] = []
        for _receiver in range(N_RECEIVERS):
            theta, theta2 = rng.uniform(0.0, 2.0 * math.pi, size=2)
            phase, phase2 = rng.uniform(0.0, 2.0 * math.pi, size=2)
            projection = (
                positions[:, 0] * math.cos(theta)
                + positions[:, 1] * math.sin(theta)
            )
            projection2 = (
                positions[:, 0] * math.cos(theta2)
                + positions[:, 1] * math.sin(theta2)
            )
            field = np.sin(2.0 * math.pi * projection / 5.5 + phase)
            field += 0.45 * np.sin(2.0 * math.pi * projection2 / 3.3 + phase2)
            receiver_labels.append(
                np.where(field >= np.median(field), 1.0, -1.0)
            )
        regimes.append(receiver_labels)

    return World(
        positions=positions,
        old_labels=regimes[0],
        new_labels=regimes[1],
        spatial=SpatialHash(positions, radius=RADIUS),
    )


def labels_for(world: World, regime: str) -> list[np.ndarray]:
    if regime == "old":
        return world.old_labels
    if regime == "new":
        return world.new_labels
    raise ValueError(regime)


def make_observer(labels: list[np.ndarray], receiver: int, seed: int):
    rng = np.random.default_rng(seed)
    cache: dict[int, float] = {}

    def observe(node: int) -> float:
        node = int(node)
        if node not in cache:
            cache[node] = float(MU * labels[receiver][node] + rng.normal())
        return cache[node]

    return observe


def query(
    graph: DevelopmentalGraph,
    cue: int,
    observe,
    *,
    learn_state: bool,
) -> tuple[float, list[EdgeEligibility], int]:
    """Normal sparse query; eligibility is a side-effect of actual transport."""
    cue = int(cue)
    seen = {cue}
    first = float(observe(cue))
    total = first
    if learn_state:
        graph.update_local_state(cue, first)

    heap: list[tuple[float, int, int, int]] = []
    counter = 0
    edge_checks = 0
    transported: list[tuple[int, int, float]] = []

    def expose(source: int) -> None:
        nonlocal counter, edge_checks
        for target, strength in zip(
            graph.neighbors[source], graph.strength[source]
        ):
            if target < 0:
                continue
            edge_checks += 1
            target = int(target)
            if target not in seen:
                heapq.heappush(
                    heap,
                    (-float(strength), counter, int(source), target),
                )
                counter += 1

    expose(cue)
    while len(seen) < QUERY_BUDGET and heap:
        _, _, source, target = heapq.heappop(heap)
        if target in seen:
            continue
        seen.add(target)
        value = float(observe(target))
        total += value
        if learn_state:
            graph.update_local_state(target, value)
        transported.append((source, target, value))
        expose(target)

    prediction_sign = 1.0 if total >= 0.0 else -1.0
    trace = [
        EdgeEligibility(source, target, prediction_sign * value)
        for source, target, value in transported
    ]
    logical_work = int(len(seen) + edge_checks)
    return prediction_sign, trace, logical_work


def evaluate(
    graphs: list[DevelopmentalGraph],
    world: World,
    regime: str,
    seed: int,
    *,
    reps: int = 2,
) -> dict:
    rng = np.random.default_rng(seed)
    labels = labels_for(world, regime)
    correct = 0
    total = 0
    work: list[float] = []

    for receiver, graph in enumerate(graphs):
        for cue in range(graph.n_nodes):
            for _ in range(reps):
                prediction, _, query_work = query(
                    graph,
                    cue,
                    make_observer(
                        labels,
                        receiver,
                        int(rng.integers(1, 2**31 - 1)),
                    ),
                    learn_state=False,
                )
                correct += int(prediction == labels[receiver][cue])
                total += 1
                work.append(float(query_work))

    return {
        "accuracy": float(correct / total),
        "mean_query_work": float(np.mean(work)),
    }


def train_lifetime(
    graphs: list[DevelopmentalGraph],
    world: World,
    regime: str,
    epochs: int,
    seed: int,
    *,
    allow_growth: bool = True,
    allow_credit: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    labels = labels_for(world, regime)
    pending: list[tuple[int, list[EdgeEligibility], float]] = []
    events = 0
    query_work = 0

    candidate0 = sum(g.candidate_inspections for g in graphs)
    growth0 = sum(g.growths for g in graphs)
    prune0 = sum(g.prunes for g in graphs)
    credit0 = sum(g.credit_updates for g in graphs)

    def settle(item: tuple[int, list[EdgeEligibility], float]) -> None:
        receiver, trace, reward = item
        if allow_credit:
            graphs[receiver].apply_delayed_credit(trace, reward)

    for _ in range(int(epochs)):
        for receiver in rng.permutation(N_RECEIVERS):
            receiver = int(receiver)
            graph = graphs[receiver]
            for cue in rng.permutation(graph.n_nodes):
                cue = int(cue)
                prediction, trace, work = query(
                    graph,
                    cue,
                    make_observer(
                        labels,
                        receiver,
                        int(rng.integers(1, 2**31 - 1)),
                    ),
                    learn_state=True,
                )
                reward = (
                    1.0 if prediction == labels[receiver][cue] else -1.0
                )
                pending.append((receiver, trace, reward))
                if allow_growth:
                    graph.grow_once(cue)
                events += 1
                query_work += int(work)
                if len(pending) > REWARD_DELAY:
                    settle(pending.pop(0))

    while pending:
        settle(pending.pop(0))

    return {
        "events": int(events),
        "query_work_per_event": float(query_work / events),
        "candidate_inspections_per_event": float(
            (sum(g.candidate_inspections for g in graphs) - candidate0) / events
        ),
        "growths_per_event": float(
            (sum(g.growths for g in graphs) - growth0) / events
        ),
        "prunes_per_event": float(
            (sum(g.prunes for g in graphs) - prune0) / events
        ),
        "credit_updates_per_event": float(
            (sum(g.credit_updates for g in graphs) - credit0) / events
        ),
    }


def make_graphs(
    world: World,
    genome: DevelopmentalGenome,
    *,
    dynamic_phase: bool = True,
) -> list[DevelopmentalGraph]:
    return [
        DevelopmentalGraph(
            world.spatial,
            genome,
            degree=DEGREE,
            dynamic_phase=dynamic_phase,
        )
        for _ in range(N_RECEIVERS)
    ]


def run_one(
    raw: np.ndarray,
    n_nodes: int,
    seed: int,
    *,
    pre_epochs: int = 4,
    post_epochs: int = 5,
    allow_growth: bool = True,
    allow_credit: bool = True,
    dynamic_phase: bool = True,
) -> dict:
    genome = DevelopmentalGenome(raw)
    world = make_world(n_nodes, seed)
    graphs = make_graphs(world, genome, dynamic_phase=dynamic_phase)

    initial = evaluate(graphs, world, "old", seed + 1)
    train_old = train_lifetime(
        graphs,
        world,
        "old",
        pre_epochs,
        seed + 2,
        allow_growth=allow_growth,
        allow_credit=allow_credit,
    )
    learned_old = evaluate(graphs, world, "old", seed + 3)
    shift = evaluate(graphs, world, "new", seed + 4)
    train_new = train_lifetime(
        graphs,
        world,
        "new",
        post_epochs,
        seed + 5,
        allow_growth=allow_growth,
        allow_credit=allow_credit,
    )
    recovered = evaluate(graphs, world, "new", seed + 6)

    return {
        "initial": initial,
        "learned_old": learned_old,
        "shift": shift,
        "recovered": recovered,
        "train_old": train_old,
        "train_new": train_new,
        "active_edges": int(sum(g.active_edges for g in graphs)),
        "phenotype_bytes": int(sum(g.persistent_bytes for g in graphs)),
        "genome_bytes": int(genome.genome_bytes),
    }


def mean_stage(runs: list[dict], stage: str, key: str) -> float:
    return float(np.mean([run[stage][key] for run in runs]))


def fitness(raw: np.ndarray, seed: int = 101, n_nodes: int = 48) -> float:
    run = run_one(
        raw,
        n_nodes,
        seed,
        pre_epochs=2,
        post_epochs=3,
    )
    developmental_gain = run["learned_old"]["accuracy"] - run["initial"]["accuracy"]
    recovery_gain = run["recovered"]["accuracy"] - run["shift"]["accuracy"]
    return float(
        2.0 * developmental_gain
        + 2.5 * recovery_gain
        + 0.3 * run["recovered"]["accuracy"]
        - 0.0002 * run["train_new"]["candidate_inspections_per_event"]
    )


def evolve_genome(
    *,
    seed: int = 7,
    population: int = 14,
    generations: int = 9,
) -> tuple[np.ndarray, float]:
    """Tiny mutation-only GA over the shared rule, never over edge identities."""
    if population < 6 or generations <= 0:
        raise ValueError("population >= 6 and generations > 0 required")
    rng = np.random.default_rng(seed)
    genomes = rng.normal(0.0, 1.0, size=(population, DevelopmentalGenome.N_GENES))
    scores = np.asarray([fitness(x) for x in genomes], dtype=np.float64)

    for generation in range(generations):
        elite = genomes[np.argsort(scores)[-4:]].copy()
        children = [elite[-1].copy(), elite[-2].copy()]
        while len(children) < population:
            parent = elite[int(rng.integers(0, len(elite)))]
            children.append(parent + rng.normal(0.0, 0.35, size=parent.shape))
        genomes = np.asarray(children, dtype=np.float64)
        scores = np.asarray([fitness(x) for x in genomes], dtype=np.float64)
        print(
            f"generation {generation:02d}  best={scores.max():.6f}  "
            f"mean={scores.mean():.6f}"
        )

    winner = int(np.argmax(scores))
    return genomes[winner].copy(), float(scores[winner])


def phenotype_replay_accuracy(raw: np.ndarray, test_seeds: tuple[int, ...]) -> float:
    """Control: inherit a grown N=64 graph instead of the developmental rule."""
    genome = DevelopmentalGenome(raw)
    source_world = make_world(64, 101)
    source_graphs = make_graphs(source_world, genome)
    train_lifetime(source_graphs, source_world, "old", 4, 103)

    accuracy = []
    for seed in test_seeds:
        world = make_world(64, seed)
        graphs = make_graphs(world, genome)
        for receiver in range(N_RECEIVERS):
            graphs[receiver].neighbors[:] = source_graphs[receiver].neighbors
            graphs[receiver].strength[:] = source_graphs[receiver].strength
        accuracy.append(evaluate(graphs, world, "old", seed + 3)["accuracy"])
    return float(np.mean(accuracy))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="rerun the deterministic outer GA instead of using the banked winner",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="include N=512 and three unseen seeds; default is a faster CI receipt",
    )
    args = parser.parse_args()

    if args.evolve:
        raw, score = evolve_genome()
        print("\nEVOLVED RAW GENOME")
        print(np.array2string(raw, precision=8, separator=", "))
        print(f"fitness={score:.6f}")
    else:
        raw = CANONICAL_RAW.copy()

    genome = DevelopmentalGenome(raw)
    print("DifferentMachine Gate 1F-A — GENOME, NOT GRAPH")
    print(
        "A 10-gene shared rule grows/prunes a fixed-degree phenotype from geometry. "
        "Normal edge use leaves eligibility; delayed scalar reward acts on that residue."
    )
    print(
        f"genome bytes={genome.genome_bytes}; evolution training size N=48; "
        "no node ids or per-edge weights are inherited"
    )
    print("decoded genome:")
    for key, value in genome.decoded().items():
        print(f"  {key:18s} {value:+.6f}")
    print()

    sizes = (64, 128, 256, 512) if args.full else (64, 128, 256)
    seeds = (41, 42, 43) if args.full else (41, 42)
    by_n: dict[int, list[dict]] = {}

    print("ERASE PHENOTYPE -> REGROW ON UNSEEN WORLD/SCALE -> HIDDEN REGIME SWITCH")
    print(
        "N    initial  learned-old  shift0  recovered  queryW  candidateW  "
        "edges/node  phenotypeKB"
    )
    for n_nodes in sizes:
        runs = [run_one(raw, n_nodes, seed) for seed in seeds]
        by_n[n_nodes] = runs
        initial = mean_stage(runs, "initial", "accuracy")
        learned = mean_stage(runs, "learned_old", "accuracy")
        shift = mean_stage(runs, "shift", "accuracy")
        recovered = mean_stage(runs, "recovered", "accuracy")
        query_work = mean_stage(runs, "recovered", "mean_query_work")
        candidate_work = float(
            np.mean(
                [
                    run["train_new"]["candidate_inspections_per_event"]
                    for run in runs
                ]
            )
        )
        edges_per_node = float(
            np.mean([run["active_edges"] / (N_RECEIVERS * n_nodes) for run in runs])
        )
        phenotype_kb = float(np.mean([run["phenotype_bytes"] for run in runs])) / 1024.0
        print(
            f"{n_nodes:4d}   {initial:7.3f}    {learned:7.3f}   {shift:7.3f}    "
            f"{recovered:7.3f}   {query_work:6.2f}     {candidate_work:6.2f}       "
            f"{edges_per_node:6.3f}       {phenotype_kb:7.1f}"
        )

    ref_n = 128
    control_seeds = seeds
    full_runs = by_n[ref_n]
    no_credit = [
        run_one(raw, ref_n, seed, allow_credit=False) for seed in control_seeds
    ]
    no_growth = [
        run_one(raw, ref_n, seed, allow_growth=False) for seed in control_seeds
    ]
    static_phase = [
        run_one(raw, ref_n, seed, dynamic_phase=False) for seed in control_seeds
    ]
    replay = phenotype_replay_accuracy(raw, tuple(control_seeds))
    rebuild_64 = mean_stage(by_n[64], "learned_old", "accuracy")

    print()
    print(f"CONTROLS AT N={ref_n}")
    print("condition                     learned-old   recovered")
    print(
        f"full developmental genome        {mean_stage(full_runs, 'learned_old', 'accuracy'):.3f}         "
        f"{mean_stage(full_runs, 'recovered', 'accuracy'):.3f}"
    )
    print(
        f"no delayed flow credit           {mean_stage(no_credit, 'learned_old', 'accuracy'):.3f}         "
        f"{mean_stage(no_credit, 'recovered', 'accuracy'):.3f}"
    )
    print(
        f"no growth                        {mean_stage(no_growth, 'learned_old', 'accuracy'):.3f}         "
        f"{mean_stage(no_growth, 'recovered', 'accuracy'):.3f}"
    )
    print(
        f"static phase, matched offset     {mean_stage(static_phase, 'learned_old', 'accuracy'):.3f}         "
        f"{mean_stage(static_phase, 'recovered', 'accuracy'):.3f}"
    )
    print()
    print("GENOME VS PHENOTYPE TRANSFER AT N=64")
    print(f"erase graph + regrow from genome : {rebuild_64:.3f}")
    print(f"replay inherited grown graph     : {replay:.3f}")
    print()

    for n_nodes, runs in by_n.items():
        assert mean_stage(runs, "learned_old", "accuracy") > mean_stage(
            runs, "initial", "accuracy"
        ) + 0.04
        assert mean_stage(runs, "recovered", "accuracy") > mean_stage(
            runs, "shift", "accuracy"
        ) + 0.015
        assert mean_stage(runs, "recovered", "mean_query_work") < 12.0
    candidate_work = [
        float(
            np.mean(
                [run["train_new"]["candidate_inspections_per_event"] for run in by_n[n]]
            )
        )
        for n in sizes
    ]
    assert max(candidate_work) / min(candidate_work) < 1.6
    assert mean_stage(full_runs, "recovered", "accuracy") > mean_stage(
        no_credit, "recovered", "accuracy"
    ) + 0.025
    assert mean_stage(full_runs, "recovered", "accuracy") > mean_stage(
        no_growth, "recovered", "accuracy"
    ) + 0.025
    assert rebuild_64 > replay + 0.08

    print("PASS — NARROW INTERPRETATION")
    print(
        "A constant-size shared developmental rule can be evolved on a small world, "
        "then grow and re-grow sparse receiver-local phenotypes on unseen worlds and "
        "larger substrates. Candidate discovery comes from constant-density geometry; "
        "normal transported signal leaves reward-modulated eligibility without Gate "
        "1E's counterfactual receiver probes."
    )
    print()
    print("GUARDRAIL")
    print(
        "This is an evo-devo / structural-plasticity mechanism receipt, not a novelty "
        "claim or a full AI architecture result. The synthetic task has smooth spatial "
        "structure, evolution used one small training world, nodes do not yet divide or "
        "die, and fixed-degree/query-budget execution makes bounded query work possible "
        "by construction. Strong ANN/router/event-RNN/MoE controls and a real stream are "
        "still required."
    )


if __name__ == "__main__":
    main()
