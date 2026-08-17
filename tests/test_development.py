import numpy as np

from different_machine.development import (
    DevelopmentalGenome,
    DevelopmentalGraph,
    EdgeEligibility,
    SpatialHash,
)


RAW = np.asarray(
    [-0.31887550, 0.00685164, 0.69949635, -0.28542975, -1.14091092,
     -1.51895493, -0.69230545, 0.44838466, 1.39977894, 0.59710920]
)


def test_genome_is_constant_size_and_contains_no_topology():
    genome = DevelopmentalGenome(RAW)
    assert genome.genome_bytes == 10 * 8
    assert not hasattr(genome, "neighbors")


def test_spatial_hash_generates_candidates_from_geometry():
    positions = np.asarray([[0.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
    spatial = SpatialHash(positions, radius=1.0)
    nearby, inspections = spatial.nearby(0)
    assert [node for node, _ in nearby] == [1]
    assert inspections >= 2


def test_growth_uses_geometry_not_a_proposal_matrix():
    genome = DevelopmentalGenome(RAW)
    positions = np.asarray([[0.0, 0.0], [0.3, 0.0], [5.0, 0.0]])
    graph = DevelopmentalGraph(SpatialHash(positions, radius=1.0), genome, degree=2)
    graph.activity[:] = 0.8
    graph.grow_once(0)
    active = set(int(x) for x in graph.neighbors[0] if x >= 0)
    assert active <= {1}
    assert 2 not in active
    assert not hasattr(graph, "proposals")


def test_delayed_credit_touches_only_edges_that_carried_signal():
    genome = DevelopmentalGenome(RAW)
    positions = np.asarray([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]])
    graph = DevelopmentalGraph(SpatialHash(positions, radius=1.0), genome, degree=2)
    graph.neighbors[0] = [1, 2]
    graph.strength[0] = [0.5, 0.5]
    before = graph.strength.copy()

    graph.apply_delayed_credit([EdgeEligibility(0, 1, 1.0)], reward=1.0)

    assert graph.strength[0, 0] != before[0, 0]
    assert np.all(graph.strength[1:] == before[1:])
    assert graph.row_updates == 1
    assert graph.credit_updates == 1
