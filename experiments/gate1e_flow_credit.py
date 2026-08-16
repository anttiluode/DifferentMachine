from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from different_machine.flow_credit import (
    FlowCreditGraph,
    FlowEligibilityTrace,
    consequence_credit,
)
from different_machine.plasticity import make_partitions

SIZES = (64, 128, 256)
SEEDS = (41, 42)
GROUP_SIZE = 8
N_RECEIVERS = 2
DEGREE = 4
PROPOSAL_DEGREE = 12
RELATIONS_PER_CONTEXT = 2
QUERY_BUDGET = 6
MU = 0.55
DECAY = 0.985
PRE_EPOCHS = 18
POST_EPOCHS = 40
REWARD_DELAY = 8
LR = 0.28
INSERT_THRESHOLD = 0.012
EVAL_REPS = 4


@dataclass
class World:
    old_groups: list[list[np.ndarray]]
    old_group_of: list[np.ndarray]
    new_groups: list[list[np.ndarray]]
    new_group_of: list[np.ndarray]
    labels: tuple[list[np.ndarray], list[np.ndarray]]
    proposals: np.ndarray


def build_world(n_nodes: int, seed: int) -> World:
    old_groups, old_group_of = make_partitions(
        n_nodes, GROUP_SIZE, N_RECEIVERS, seed=seed + 11, aligned=False
    )
    new_groups, new_group_of = make_partitions(
        n_nodes, GROUP_SIZE, N_RECEIVERS, seed=seed + 10011, aligned=False
    )
    rng = np.random.default_rng(seed + 22222)

    labels = []
    for _ in range(2):
        labels.append(
            [
                rng.choice(
                    np.asarray([-1.0, 1.0]), size=n_nodes // GROUP_SIZE
                )
                for _ in range(N_RECEIVERS)
            ]
        )

    proposals = np.empty((n_nodes, PROPOSAL_DEGREE), dtype=np.int64)
    for i in range(n_nodes):
        candidates: list[int] = []
        # Deliberately privileged proposal scaffold: useful old/new relations
        # for both receivers are locally available, mixed with distractors.
        # Plasticity sees candidate ids only, never which hidden relation made
        # a candidate useful.
        for groups_all, group_of_all in (
            (old_groups, old_group_of),
            (new_groups, new_group_of),
        ):
            for r in range(N_RECEIVERS):
                members = groups_all[r][int(group_of_all[r][i])]
                options = members[members != i]
                for x in rng.choice(
                    options, size=RELATIONS_PER_CONTEXT, replace=False
                ):
                    xx = int(x)
                    if xx not in candidates:
                        candidates.append(xx)

        while len(candidates) < PROPOSAL_DEGREE:
            j = int(rng.integers(0, n_nodes))
            if j != i and j not in candidates:
                candidates.append(j)
        proposals[i] = np.asarray(
            candidates[:PROPOSAL_DEGREE], dtype=np.int64
        )

    return World(
        old_groups=old_groups,
        old_group_of=old_group_of,
        new_groups=new_groups,
        new_group_of=new_group_of,
        labels=(labels[0], labels[1]),
        proposals=proposals,
    )


def regime_parts(world: World, regime: str):
    if regime == "old":
        return world.old_group_of, world.labels[0]
    if regime == "new":
        return world.new_group_of, world.labels[1]
    raise ValueError(regime)


def make_event_observer(
    receiver: int,
    group_of_all: list[np.ndarray],
    labels_all: list[np.ndarray],
    event_seed: int,
):
    cache: dict[int, float] = {}

    def observe(node: int) -> float:
        node = int(node)
        if node not in cache:
            local_seed = (
                int(event_seed) * 1000003
                + (node + 1) * 9176
                + (receiver + 1) * 6113
            ) & 0xFFFFFFFF
            rng = np.random.default_rng(local_seed)
            y = float(labels_all[receiver][int(group_of_all[receiver][node])])
            cache[node] = MU * y + float(rng.normal())
        return cache[node]

    return observe


def query(
    graph: FlowCreditGraph, cue: int, observe
) -> tuple[float, int]:
    cue = int(cue)
    seen = {cue}
    total = float(observe(cue))
    heap: list[tuple[float, int, int]] = []
    counter = 0
    edge_inspections = 0

    def expose(i: int) -> None:
        nonlocal counter, edge_inspections
        for j, strength in zip(graph.neighbors[i], graph.strength[i]):
            j = int(j)
            if j < 0:
                continue
            edge_inspections += 1
            if j not in seen:
                heapq.heappush(heap, (-float(strength), counter, j))
                counter += 1

    expose(cue)
    while len(seen) < QUERY_BUDGET and heap:
        _, _, j = heapq.heappop(heap)
        if j in seen:
            continue
        seen.add(j)
        total += float(observe(j))
        expose(j)

    return total, int(len(seen) + edge_inspections)


def measure_trace(
    graph: FlowCreditGraph,
    cue: int,
    observe,
    rng: np.random.Generator,
) -> tuple[FlowEligibilityTrace, float, int]:
    baseline, work0 = query(graph, cue, observe)
    prediction_sign = 1.0 if baseline >= 0.0 else -1.0

    # Existing-edge flow contribution: one bounded local ablation.
    slot = int(rng.integers(0, graph.degree))
    existing_neighbor = int(graph.neighbors[cue, slot])
    old_strength = float(graph.strength[cue, slot])
    graph.neighbors[cue, slot] = -1
    graph.strength[cue, slot] = 0.0
    ablated, work1 = query(graph, cue, observe)
    graph.neighbors[cue, slot] = existing_neighbor
    graph.strength[cue, slot] = old_strength
    existing_delta = baseline - ablated

    # Candidate flow contribution: one bounded before/after structural probe.
    candidate = graph.sample_candidate(cue, rng)
    weakest = int(np.argmin(graph.strength[cue]))
    displaced_neighbor = int(graph.neighbors[cue, weakest])
    displaced_strength = float(graph.strength[cue, weakest])
    graph.neighbors[cue, weakest] = candidate
    graph.strength[cue, weakest] = max(
        0.25, float(np.max(graph.strength[cue]) + 0.05)
    )
    perturbed, work2 = query(graph, cue, observe)
    graph.neighbors[cue, weakest] = displaced_neighbor
    graph.strength[cue, weakest] = displaced_strength
    candidate_delta = perturbed - baseline

    trace = FlowEligibilityTrace(
        cue=int(cue),
        existing_neighbor=existing_neighbor,
        existing_delta=float(existing_delta),
        candidate_neighbor=int(candidate),
        candidate_delta=float(candidate_delta),
        prediction_sign=float(prediction_sign),
    )
    return trace, float(baseline), int(work0 + work1 + work2)


def settle_trace(
    graph: FlowCreditGraph,
    trace: FlowEligibilityTrace,
    reward: float,
    mode: str,
) -> None:
    if mode in ("credit", "outcome_shuffle"):
        existing_credit = consequence_credit(
            reward, trace.prediction_sign, trace.existing_delta
        )
        candidate_credit = consequence_credit(
            reward, trace.prediction_sign, trace.candidate_delta
        )
    elif mode == "flow_only":
        existing_credit = abs(trace.existing_delta)
        candidate_credit = abs(trace.candidate_delta)
    elif mode == "reward_only":
        existing_credit = float(reward)
        candidate_credit = float(reward)
    elif mode == "frozen":
        return
    else:
        raise ValueError(mode)

    graph.apply_credits(
        cue=trace.cue,
        existing_neighbor=trace.existing_neighbor,
        existing_credit=float(np.clip(existing_credit, -2.0, 2.0)),
        candidate_neighbor=trace.candidate_neighbor,
        candidate_credit=float(np.clip(candidate_credit, -2.0, 2.0)),
        lr=LR,
        insert_threshold=INSERT_THRESHOLD,
    )


def train_stream(
    graphs: list[FlowCreditGraph],
    world: World,
    regime: str,
    epochs: int,
    seed: int,
    mode: str,
) -> dict:
    rng = np.random.default_rng(seed)
    group_of_all, labels_all = regime_parts(world, regime)
    pending: list[tuple[FlowEligibilityTrace, float, int]] = []
    probe_work = 0
    task_events = 0

    slot_ops0 = sum(g.slot_ops for g in graphs)
    proposal_ops0 = sum(g.proposal_ops for g in graphs)
    row_updates0 = sum(g.row_updates for g in graphs)

    def settle_oldest() -> None:
        trace, reward, receiver = pending.pop(0)
        used_reward = reward
        if mode == "outcome_shuffle" and pending:
            # Actual reward from a different nearby event: same reward alphabet,
            # broken trace/outcome pairing.
            idx = int(rng.integers(0, len(pending)))
            used_reward = float(pending[idx][1])
        settle_trace(graphs[receiver], trace, used_reward, mode)

    for _ in range(int(epochs)):
        for receiver in rng.permutation(N_RECEIVERS):
            receiver = int(receiver)
            for cue in rng.permutation(graphs[receiver].n_nodes):
                cue = int(cue)
                event_seed = int(rng.integers(1, 2**31 - 1))
                observe = make_event_observer(
                    receiver, group_of_all, labels_all, event_seed
                )
                trace, baseline, work = measure_trace(
                    graphs[receiver], cue, observe, rng
                )
                target = float(
                    labels_all[receiver][int(group_of_all[receiver][cue])]
                )
                prediction = 1.0 if baseline >= 0.0 else -1.0
                reward = 1.0 if prediction == target else -1.0
                pending.append((trace, reward, receiver))
                probe_work += int(work)
                task_events += 1
                if len(pending) > REWARD_DELAY:
                    settle_oldest()

    while pending:
        settle_oldest()

    slot_ops = sum(g.slot_ops for g in graphs) - slot_ops0
    proposal_ops = sum(g.proposal_ops for g in graphs) - proposal_ops0
    row_updates = sum(g.row_updates for g in graphs) - row_updates0
    return {
        "events": int(task_events),
        "probe_query_work_per_event": float(probe_work / task_events),
        "slot_ops_per_event": float(slot_ops / task_events),
        "proposal_ops_per_event": float(proposal_ops / task_events),
        "row_updates_per_event": float(row_updates / task_events),
    }


def evaluate(
    graphs: list[FlowCreditGraph],
    world: World,
    regime: str,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    group_of_all, labels_all = regime_parts(world, regime)
    correct = 0
    total = 0
    logical_work = []

    for receiver, graph in enumerate(graphs):
        for cue in range(graph.n_nodes):
            target = float(
                labels_all[receiver][int(group_of_all[receiver][cue])]
            )
            for _ in range(EVAL_REPS):
                observe = make_event_observer(
                    receiver,
                    group_of_all,
                    labels_all,
                    int(rng.integers(1, 2**31 - 1)),
                )
                output, work = query(graph, cue, observe)
                prediction = 1.0 if output >= 0.0 else -1.0
                correct += int(prediction == target)
                total += 1
                logical_work.append(float(work))

    return {
        "accuracy": float(correct / total),
        "mean_query_work": float(np.mean(logical_work)),
    }


def run_one(n_nodes: int, seed: int, mode: str = "credit") -> dict:
    world = build_world(n_nodes, seed)
    rng = np.random.default_rng(seed + 77)
    graphs = [
        FlowCreditGraph.from_proposals(
            world.proposals, DEGREE, rng, decay=DECAY
        )
        for _ in range(N_RECEIVERS)
    ]

    initial = evaluate(graphs, world, "old", seed + 1)
    train_old = train_stream(
        graphs, world, "old", PRE_EPOCHS, seed + 2, mode
    )
    learned_old = evaluate(graphs, world, "old", seed + 3)
    shift = evaluate(graphs, world, "new", seed + 4)
    train_new = train_stream(
        graphs, world, "new", POST_EPOCHS, seed + 5, mode
    )
    recovered = evaluate(graphs, world, "new", seed + 6)

    return {
        "initial": initial,
        "learned_old": learned_old,
        "shift": shift,
        "recovered": recovered,
        "train_old": train_old,
        "train_new": train_new,
        "persistent_bytes": int(sum(g.persistent_bytes for g in graphs)),
    }


def mean_metric(runs: list[dict], stage: str, key: str) -> float:
    return float(np.mean([run[stage][key] for run in runs]))


def main() -> None:
    print("DifferentMachine Gate 1E — DELAYED TASK CONSEQUENCE x CAUSAL FLOW")
    print(
        "No useful pair is supplied to plasticity. One existing-edge flow effect "
        "and one candidate before/after flow effect are stored; task outcome "
        "(+1 correct / -1 wrong) arrives 8 events later."
    )
    print(
        "credit = delayed outcome x receiver's own earlier output sign x flow delta"
    )
    print()

    by_n: dict[int, list[dict]] = defaultdict(list)
    for n in SIZES:
        for seed in SEEDS:
            by_n[n].append(run_one(n, seed, mode="credit"))

    print("TASK-DRIVEN LEARNING -> HIDDEN REGIME SWITCH -> RELEARNING")
    print("N    initial   learned-old   shift0   recovered   query-work   KB")
    learned_acc = {}
    recovered_acc = {}
    for n in SIZES:
        runs = by_n[n]
        initial = mean_metric(runs, "initial", "accuracy")
        old = mean_metric(runs, "learned_old", "accuracy")
        shift = mean_metric(runs, "shift", "accuracy")
        rec = mean_metric(runs, "recovered", "accuracy")
        qwork = mean_metric(runs, "recovered", "mean_query_work")
        kb = float(np.mean([r["persistent_bytes"] for r in runs])) / 1024.0
        learned_acc[n] = old
        recovered_acc[n] = rec
        print(
            f"{n:4d}   {initial:7.3f}      {old:7.3f}    {shift:7.3f}     "
            f"{rec:7.3f}       {qwork:7.1f}   {kb:5.1f}"
        )

    ref_n = 128
    print()
    print(f"CAUSAL CREDIT CONTROLS AT N={ref_n}")
    print("mode                 learned-old   recovered")
    controls = {}
    for mode in ("outcome_shuffle", "flow_only", "reward_only", "frozen"):
        runs = [run_one(ref_n, seed, mode=mode) for seed in SEEDS]
        controls[mode] = runs
        print(
            f"{mode:20s}   {mean_metric(runs, 'learned_old', 'accuracy'):7.3f}      "
            f"{mean_metric(runs, 'recovered', 'accuracy'):7.3f}"
        )

    main_ref_new = mean_metric(by_n[ref_n], "recovered", "accuracy")
    ctl_best_new = max(
        mean_metric(runs, "recovered", "accuracy")
        for runs in controls.values()
    )

    assert all(x > 0.80 for x in learned_acc.values())
    assert all(x > 0.80 for x in recovered_acc.values())
    assert main_ref_new > ctl_best_new + 0.10

    receipt = by_n[ref_n][0]["train_new"]
    print()
    print("BOUNDED LEARNING RECEIPT (representative N=128 run)")
    print(
        f"probe query work/event {receipt['probe_query_work_per_event']:.1f}; "
        f"proposal inspections/event {receipt['proposal_ops_per_event']:.1f}; "
        f"slot ops/event {receipt['slot_ops_per_event']:.1f}; "
        f"row updates/event {receipt['row_updates_per_event']:.2f}"
    )
    print()
    print("PASS — NARROW INTERPRETATION")
    print(
        "Given a bounded local proposal scaffold that contains useful alternatives, "
        "receiver-relative topology can be selected and re-selected from delayed "
        "end-to-end task consequence without a useful-pair teaching event."
    )
    print()
    print("GUARDRAIL")
    print(
        "This does not solve proposal discovery. The 12-candidate scaffold is built "
        "so old/new relationships for both receivers are locally available, and the "
        "before/after flow instrument spends two extra bounded queries per task event. "
        "The binary task also makes success/failure unusually informative. Gate 1E "
        "therefore removes the pair-label cheat while leaving proposal-scaffold, "
        "perturbation-cost and task-simplicity cheats for the next stage."
    )


if __name__ == "__main__":
    main()
