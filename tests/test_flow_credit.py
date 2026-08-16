import numpy as np

from different_machine.flow_credit import FlowCreditGraph, consequence_credit


def test_consequence_credit_uses_delayed_outcome_and_flow_sign():
    assert consequence_credit(+1.0, +1.0, +0.5) == 0.5
    assert consequence_credit(-1.0, +1.0, +0.5) == -0.5
    assert consequence_credit(+1.0, -1.0, +0.5) == -0.5


def test_credit_update_touches_only_addressed_row_and_can_insert_candidate():
    proposals = np.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [0, 2, 3, 4, 5, 6],
            [0, 1, 3, 4, 5, 6],
            [0, 1, 2, 4, 5, 6],
            [0, 1, 2, 3, 5, 6],
            [0, 1, 2, 3, 4, 6],
            [0, 1, 2, 3, 4, 5],
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(7)
    graph = FlowCreditGraph.from_proposals(proposals, degree=3, rng=rng)
    cue = 2
    before_neighbors = graph.neighbors.copy()
    before_strength = graph.strength.copy()
    existing = int(graph.neighbors[cue, 0])
    candidate = next(
        int(x) for x in proposals[cue] if x not in set(graph.neighbors[cue])
    )

    graph.apply_credits(
        cue=cue,
        existing_neighbor=existing,
        existing_credit=0.5,
        candidate_neighbor=candidate,
        candidate_credit=0.8,
        lr=0.3,
        insert_threshold=0.01,
    )

    changed_rows = set(
        np.flatnonzero(np.any(graph.neighbors != before_neighbors, axis=1))
    )
    assert changed_rows <= {cue}
    assert candidate in set(graph.neighbors[cue])
    untouched = [i for i in range(graph.n_nodes) if i != cue]
    assert np.array_equal(
        graph.neighbors[untouched], before_neighbors[untouched]
    )
    assert np.array_equal(graph.strength[untouched], before_strength[untouched])
    assert graph.row_updates == 1
    assert graph.persistent_bytes == (
        proposals.nbytes + graph.neighbors.nbytes + graph.strength.nbytes
    )
