from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from different_machine.online_plasticity import (
    OnlineLocalGraph,
    RecentLocalGraph,
    sample_receiver_couse_pair,
)
from different_machine.plasticity import evaluate_graph, make_partitions


SIZES = (64, 128, 256, 512)
SEEDS = (41, 42, 43)
GROUP_SIZE = 8
DEGREE = 4
N_RECEIVERS = 2
SIGNAL = 0.80
PRE_EVENTS_PER_NODE = 60
DECAY = 0.97
CHECKPOINTS = (0, 8, 16, 24, 32, 48, 64)
TARGET_RECALL = 0.75


def fit_alpha(n_to_work: dict[int, float]) -> float:
    ns = np.asarray(sorted(n_to_work), dtype=np.float64)
    work = np.asarray([n_to_work[int(n)] for n in ns], dtype=np.float64)
    return float(np.polyfit(np.log(ns), np.log(work), 1)[0])


def mean_receiver_work(graphs, groups_all, group_of_all) -> tuple[float, float]:
    metrics = [
        evaluate_graph(
            graphs[r],
            groups_all[r],
            group_of_all[r],
            target_recall=TARGET_RECALL,
        )
        for r in range(N_RECEIVERS)
    ]
    return (
        float(np.mean([m["mean_logical_work"] for m in metrics])),
        float(np.mean([m["success_rate"] for m in metrics])),
    )


def feed_events(graph, groups_all, group_of_all, receiver, rng, n_events) -> None:
    for _ in range(int(n_events)):
        i, j = sample_receiver_couse_pair(
            groups_all,
            group_of_all,
            receiver=receiver,
            rng=rng,
            signal_probability=SIGNAL,
        )
        graph.update_pair(i, j)


def run_online_one(n_nodes: int, seed: int) -> dict:
    old_groups, old_group_of = make_partitions(
        n_nodes, GROUP_SIZE, N_RECEIVERS, seed=seed + 100, aligned=False
    )
    new_groups, new_group_of = make_partitions(
        n_nodes, GROUP_SIZE, N_RECEIVERS, seed=seed + 10000, aligned=False
    )
    graphs = [
        OnlineLocalGraph.empty(n_nodes, degree=DEGREE, decay=DECAY)
        for _ in range(N_RECEIVERS)
    ]
    rng = np.random.default_rng(seed + 50000)

    for r in range(N_RECEIVERS):
        feed_events(
            graphs[r], old_groups, old_group_of, r, rng,
            PRE_EVENTS_PER_NODE * n_nodes,
        )

    steady_work, steady_success = mean_receiver_work(
        graphs, old_groups, old_group_of
    )

    curve = {}
    previous = 0
    for checkpoint in CHECKPOINTS:
        delta = checkpoint - previous
        if delta:
            for r in range(N_RECEIVERS):
                feed_events(
                    graphs[r], new_groups, new_group_of, r, rng,
                    delta * n_nodes,
                )
        work, success = mean_receiver_work(graphs, new_groups, new_group_of)
        curve[int(checkpoint)] = {"work": work, "success": success}
        previous = checkpoint

    total_pair_events = N_RECEIVERS * (PRE_EVENTS_PER_NODE + CHECKPOINTS[-1]) * n_nodes
    return {
        "steady_work": steady_work,
        "steady_success": steady_success,
        "curve": curve,
        "persistent_bytes": int(sum(g.persistent_bytes for g in graphs)),
        "slot_ops_per_pair": float(sum(g.slot_ops for g in graphs) / total_pair_events),
        "row_updates_per_pair": float(sum(g.row_updates for g in graphs) / total_pair_events),
    }


def run_no_forgetting(n_nodes: int, seed: int, post_events_per_node: int = 96) -> float:
    old_groups, old_group_of = make_partitions(
        n_nodes, GROUP_SIZE, N_RECEIVERS, seed=seed + 100, aligned=False
    )
    new_groups, new_group_of = make_partitions(
        n_nodes, GROUP_SIZE, N_RECEIVERS, seed=seed + 10000, aligned=False
    )
    graphs = [
        OnlineLocalGraph.empty(n_nodes, degree=DEGREE, decay=1.0)
        for _ in range(N_RECEIVERS)
    ]
    rng = np.random.default_rng(seed + 50000)

    for r in range(N_RECEIVERS):
        feed_events(
            graphs[r], old_groups, old_group_of, r, rng,
            PRE_EVENTS_PER_NODE * n_nodes,
        )
        feed_events(
            graphs[r], new_groups, new_group_of, r, rng,
            post_events_per_node * n_nodes,
        )

    work, _ = mean_receiver_work(graphs, new_groups, new_group_of)
    return work


def run_recency_control(n_nodes: int, seed: int, post_events_per_node: int = 48) -> float:
    old_groups, old_group_of = make_partitions(
        n_nodes, GROUP_SIZE, N_RECEIVERS, seed=seed + 100, aligned=False
    )
    new_groups, new_group_of = make_partitions(
        n_nodes, GROUP_SIZE, N_RECEIVERS, seed=seed + 10000, aligned=False
    )
    graphs = [RecentLocalGraph.empty(n_nodes, degree=DEGREE) for _ in range(N_RECEIVERS)]
    rng = np.random.default_rng(seed + 50000)

    for r in range(N_RECEIVERS):
        feed_events(
            graphs[r], old_groups, old_group_of, r, rng,
            PRE_EVENTS_PER_NODE * n_nodes,
        )
        feed_events(
            graphs[r], new_groups, new_group_of, r, rng,
            post_events_per_node * n_nodes,
        )

    work, _ = mean_receiver_work(graphs, new_groups, new_group_of)
    return work


def main() -> None:
    print("DifferentMachine Gate 1D — BOUNDED ONLINE LOCAL PLASTICITY")
    print(
        "degree=4, two receiver-local overlays, no global pair table, "
        "60 old-regime events/node/receiver"
    )
    print(
        "At each pair event only the two endpoint rows are updated. "
        "Rows age only when locally touched."
    )
    print()

    all_results = defaultdict(list)
    for n in SIZES:
        for seed in SEEDS:
            all_results[n].append(run_online_one(n, seed))

    print("STEADY OLD REGIME -> HIDDEN PARTITION CHANGE -> ONLINE REACQUISITION")
    print("N    steady   shift0    e8      e16     e24     e32     e48     e64    KB   slotOps/pair")
    steady_by_n = {}
    shift_by_n = {}
    recovered_by_n = {}

    for n in SIZES:
        runs = all_results[n]
        steady = float(np.mean([x["steady_work"] for x in runs]))
        vals = {
            cp: float(np.mean([x["curve"][cp]["work"] for x in runs]))
            for cp in CHECKPOINTS
        }
        kb = float(np.mean([x["persistent_bytes"] for x in runs])) / 1024.0
        slot_ops = float(np.mean([x["slot_ops_per_pair"] for x in runs]))
        steady_by_n[n] = steady
        shift_by_n[n] = vals[0]
        recovered_by_n[n] = vals[48]
        print(
            f"{n:4d}  {steady:7.1f}  {vals[0]:7.1f}  {vals[8]:7.1f}  "
            f"{vals[16]:7.1f}  {vals[24]:7.1f}  {vals[32]:7.1f}  "
            f"{vals[48]:7.1f}  {vals[64]:7.1f}  {kb:4.0f}      {slot_ops:6.2f}"
        )

    alpha_steady = fit_alpha(steady_by_n)
    alpha_shift = fit_alpha(shift_by_n)
    alpha_recovered = fit_alpha(recovered_by_n)
    print()
    print(
        f"fit logical query work: steady ~ N^{alpha_steady:.3f}, "
        f"immediately-after-change ~ N^{alpha_shift:.3f}, "
        f"after 48 new events/node ~ N^{alpha_recovered:.3f}"
    )
    print()

    print("CONTROLS")
    print("N    no-forgetting@96/node   recency-only@48/node   online-decay@48/node")
    no_forget_by_n = {}
    recent_by_n = {}
    for n in SIZES:
        no_forget = float(np.mean([run_no_forgetting(n, s) for s in SEEDS]))
        recent = float(np.mean([run_recency_control(n, s) for s in SEEDS]))
        online = recovered_by_n[n]
        no_forget_by_n[n] = no_forget
        recent_by_n[n] = recent
        print(f"{n:4d}        {no_forget:9.1f}              {recent:9.1f}              {online:9.1f}")

    print(
        "control fits: "
        f"no-forgetting ~ N^{fit_alpha(no_forget_by_n):.3f}, "
        f"recency-only ~ N^{fit_alpha(recent_by_n):.3f}"
    )
    print()

    # Broad deterministic gate assertions, intentionally not exact-number locks.
    assert alpha_steady < 0.10
    assert alpha_recovered < 0.10
    assert 0.75 < alpha_shift < 1.20
    assert all(recovered_by_n[n] < 50.0 for n in SIZES)
    assert all(no_forget_by_n[n] > 2.5 * recovered_by_n[n] for n in SIZES)
    assert all(recent_by_n[n] > 2.5 * recovered_by_n[n] for n in SIZES)

    print("PASS — NARROW INTERPRETATION")
    print(
        "A fixed-degree receiver-local graph can be learned and re-learned online "
        "from pair experience with O(degree) endpoint updates. In this synthetic "
        "world, useful query work is approximately capacity-independent before a "
        "regime change, blows up toward global-search scaling when the learned "
        "relationship becomes wrong, and returns to approximately flat scaling "
        "after about 32-48 new events per node."
    )
    print()
    print("GUARDRAIL")
    print(
        "This is a bounded decayed associative/plasticity heuristic on a known-answer "
        "co-use world, not a new learning theorem or a full DifferentMachine result. "
        "The next gate must use end-to-end task loss, online receiver/task switching, "
        "and strong sparse retrieval / ANN / delta / event-RNN / MoE controls with "
        "actual update+query wall clock and memory traffic."
    )


if __name__ == "__main__":
    main()
