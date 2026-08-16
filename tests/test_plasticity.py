from different_machine.plasticity import (
    collect_receiver_couse_counts,
    compile_topk_graph,
    evaluate_graph,
    make_partitions,
)


def test_aligned_partitions_are_identical():
    groups, group_of = make_partitions(64, 8, 2, seed=7, aligned=True)
    assert (group_of[0] == group_of[1]).all()
    for a, b in zip(groups[0], groups[1]):
        assert (a == b).all()


def test_compiled_graph_has_bounded_degree_and_expected_bytes():
    receiver_counts, _, groups, group_of, _ = collect_receiver_couse_counts(
        n_nodes=64,
        group_size=8,
        n_receivers=2,
        samples_per_node=40,
        signal_probability=0.8,
        seed=11,
        aligned=False,
    )
    graph = compile_topk_graph(receiver_counts[0], degree=4)
    assert graph.neighbors.shape == (64, 4)
    assert graph.strength.shape == (64, 4)
    assert graph.persistent_bytes == 64 * 4 * 16

    result = evaluate_graph(graph, groups[0], group_of[0], target_recall=0.75)
    assert result["success_rate"] > 0.95
    assert result["mean_touches"] < 12.0


def test_receiver_specific_graph_beats_pooled_under_conflict():
    receiver_counts, pooled_counts, groups, group_of, _ = collect_receiver_couse_counts(
        n_nodes=128,
        group_size=8,
        n_receivers=2,
        samples_per_node=60,
        signal_probability=0.8,
        seed=1234,
        aligned=False,
    )
    pooled = compile_topk_graph(pooled_counts, degree=4)
    receiver = compile_topk_graph(receiver_counts[0], degree=4)

    p = evaluate_graph(pooled, groups[0], group_of[0], target_recall=0.75)
    r = evaluate_graph(receiver, groups[0], group_of[0], target_recall=0.75)
    assert r["mean_logical_work"] < p["mean_logical_work"]
