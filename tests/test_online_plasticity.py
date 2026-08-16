import numpy as np

from different_machine.online_plasticity import OnlineLocalGraph, sample_receiver_couse_pair
from different_machine.plasticity import evaluate_graph, make_partitions


def test_pair_update_touches_only_two_rows_and_memory_is_fixed():
    graph = OnlineLocalGraph.empty(32, degree=4, decay=0.97)
    before_neighbors = graph.neighbors.copy()
    before_strength = graph.strength.copy()

    graph.update_pair(3, 11)

    changed_rows = set(np.flatnonzero(np.any(graph.neighbors != before_neighbors, axis=1)))
    assert changed_rows == {3, 11}
    assert graph.row_updates == 2
    assert graph.persistent_bytes == 32 * 4 * (8 + 8)
    assert graph.slot_ops <= 8 * graph.degree
    assert np.all(graph.neighbors[[i for i in range(32) if i not in changed_rows]] == -1)
    assert np.all(graph.strength[[i for i in range(32) if i not in changed_rows]] == before_strength[[i for i in range(32) if i not in changed_rows]])


def test_repeated_pair_reinforces_existing_local_edge():
    graph = OnlineLocalGraph.empty(16, degree=4, decay=0.97)
    graph.update_pair(2, 7)
    slot = int(np.flatnonzero(graph.neighbors[2] == 7)[0])
    s0 = float(graph.strength[2, slot])
    graph.update_pair(2, 7)
    s1 = float(graph.strength[2, slot])
    assert s1 > s0
    assert graph.reinforcements >= 2


def test_local_plasticity_reacquires_after_regime_change():
    n = 64
    group_size = 8
    receivers = 2
    old_groups, old_group_of = make_partitions(n, group_size, receivers, seed=31)
    new_groups, new_group_of = make_partitions(n, group_size, receivers, seed=1031)
    graphs = [OnlineLocalGraph.empty(n, degree=4, decay=0.97) for _ in range(receivers)]
    rng = np.random.default_rng(99)

    def feed(groups, group_of, events_per_node):
        for r in range(receivers):
            for _ in range(events_per_node * n):
                i, j = sample_receiver_couse_pair(groups, group_of, r, rng, 0.80)
                graphs[r].update_pair(i, j)

    feed(old_groups, old_group_of, 60)
    old_work = np.mean([
        evaluate_graph(graphs[r], old_groups[r], old_group_of[r])["mean_logical_work"]
        for r in range(receivers)
    ])
    new_work_before = np.mean([
        evaluate_graph(graphs[r], new_groups[r], new_group_of[r])["mean_logical_work"]
        for r in range(receivers)
    ])

    feed(new_groups, new_group_of, 48)
    new_work_after = np.mean([
        evaluate_graph(graphs[r], new_groups[r], new_group_of[r])["mean_logical_work"]
        for r in range(receivers)
    ])

    assert old_work < 45.0
    assert new_work_before > 100.0
    assert new_work_after < 45.0
