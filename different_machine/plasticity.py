from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


CountRow = Dict[int, int]


@dataclass
class LocalGraph:
    """A bounded-degree persistent adjacency overlay.

    `neighbors[i]` are the only destinations that become locally discoverable
    when node i is touched. `strength[i]` is learned association strength used
    only to prioritize the local frontier.
    """

    neighbors: np.ndarray
    strength: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.neighbors.shape[0])

    @property
    def degree(self) -> int:
        return int(self.neighbors.shape[1])

    @property
    def persistent_bytes(self) -> int:
        return int(self.neighbors.nbytes + self.strength.nbytes)


def make_partitions(
    n_nodes: int,
    group_size: int,
    n_receivers: int,
    seed: int,
    aligned: bool = False,
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    """Create receiver-specific latent 'things that matter together' partitions.

    Group labels are used only by the evaluator. The learner never receives
    them; it sees pairwise co-use events.
    """
    if n_nodes % group_size:
        raise ValueError("n_nodes must be divisible by group_size")

    rng = np.random.default_rng(seed)
    groups_all: List[List[np.ndarray]] = []
    group_of_all: List[np.ndarray] = []

    first_perm = rng.permutation(n_nodes)
    for r in range(n_receivers):
        perm = first_perm.copy() if (aligned and r > 0) else (
            first_perm if r == 0 else rng.permutation(n_nodes)
        )
        groups: List[np.ndarray] = []
        group_of = np.empty(n_nodes, dtype=np.int64)
        for g, start in enumerate(range(0, n_nodes, group_size)):
            members = np.asarray(perm[start : start + group_size], dtype=np.int64)
            groups.append(members)
            group_of[members] = g
        groups_all.append(groups)
        group_of_all.append(group_of)

    return groups_all, group_of_all


def _add_pair(rows: Sequence[CountRow], i: int, j: int) -> None:
    if i == j:
        return
    rows[i][j] = rows[i].get(j, 0) + 1
    rows[j][i] = rows[j].get(i, 0) + 1


def collect_receiver_couse_counts(
    n_nodes: int,
    group_size: int = 8,
    n_receivers: int = 2,
    samples_per_node: int = 60,
    signal_probability: float = 0.80,
    seed: int = 0,
    aligned: bool = False,
) -> Tuple[
    List[List[CountRow]],
    List[CountRow],
    List[List[np.ndarray]],
    List[np.ndarray],
    int,
]:
    """Observe receiver-labelled pairwise co-use without exposing group labels.

    For receiver r, most pairs come from r's latent partition. Structured
    distractors come from another receiver's partition rather than from uniform
    noise. In the conflicting condition this makes a pooled/global notion of
    locality genuinely ambiguous.

    The count table is a deliberately generous *learning-time* instrument. It is
    compiled away after training; an online bounded-memory plasticity rule is a
    later gate.
    """
    if not 0.0 <= signal_probability <= 1.0:
        raise ValueError("signal_probability must be in [0, 1]")

    groups_all, group_of_all = make_partitions(
        n_nodes=n_nodes,
        group_size=group_size,
        n_receivers=n_receivers,
        seed=seed + 1,
        aligned=aligned,
    )
    receiver_counts: List[List[CountRow]] = [
        [defaultdict(int) for _ in range(n_nodes)] for _ in range(n_receivers)
    ]
    pooled_counts: List[CountRow] = [defaultdict(int) for _ in range(n_nodes)]

    rng = np.random.default_rng(seed + 2)
    total_events = 0
    for r in range(n_receivers):
        other = (r + 1) % n_receivers if n_receivers > 1 else r
        for _ in range(samples_per_node * n_nodes):
            i = int(rng.integers(0, n_nodes))
            source_r = r if rng.random() < signal_probability else other
            members = groups_all[source_r][group_of_all[source_r][i]]
            candidates = members[members != i]
            j = int(rng.choice(candidates))

            _add_pair(receiver_counts[r], i, j)
            _add_pair(pooled_counts, i, j)
            total_events += 1

    return receiver_counts, pooled_counts, groups_all, group_of_all, total_events


def compile_topk_graph(counts: Sequence[CountRow], degree: int) -> LocalGraph:
    """Compile pairwise experience into a bounded-degree local substrate."""
    n_nodes = len(counts)
    neighbors = np.full((n_nodes, degree), -1, dtype=np.int64)
    strength = np.zeros((n_nodes, degree), dtype=np.float64)

    for i, row in enumerate(counts):
        top = sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))[:degree]
        for slot, (j, count) in enumerate(top):
            neighbors[i, slot] = int(j)
            strength[i, slot] = float(count)

    return LocalGraph(neighbors=neighbors, strength=strength)


def random_graph(n_nodes: int, degree: int, seed: int = 0) -> LocalGraph:
    rng = np.random.default_rng(seed)
    neighbors = np.empty((n_nodes, degree), dtype=np.int64)
    strength = np.ones((n_nodes, degree), dtype=np.float64)
    all_nodes = np.arange(n_nodes)
    for i in range(n_nodes):
        pool = all_nodes[all_nodes != i]
        neighbors[i] = rng.choice(pool, size=degree, replace=False)
    return LocalGraph(neighbors=neighbors, strength=strength)


def query_group_recall(
    graph: LocalGraph,
    groups: Sequence[np.ndarray],
    group_of: np.ndarray,
    cue: int,
    target_recall: float = 0.75,
    max_touches: int | None = None,
) -> Tuple[int | None, int, float]:
    """Traverse only locally exposed edges until enough receiver-relevant items appear.

    Returns `(node_touches_or_None, edge_inspections, achieved_recall)`.
    One node touch exposes that node's bounded local adjacency. No global scan is
    performed by the query.
    """
    target = set(int(x) for x in groups[int(group_of[cue])])
    need = int(math.ceil(target_recall * len(target)))

    seen = {int(cue)}
    hits = 1
    touches = 1
    edge_inspections = 0
    heap: List[Tuple[float, int, int]] = []
    counter = 0

    def expose(i: int) -> None:
        nonlocal edge_inspections, counter
        for j, s in zip(graph.neighbors[i], graph.strength[i]):
            if j < 0:
                continue
            edge_inspections += 1
            if int(j) not in seen:
                heapq.heappush(heap, (-float(s), counter, int(j)))
                counter += 1

    expose(int(cue))
    while hits < need and heap and (max_touches is None or touches < max_touches):
        _, _, j = heapq.heappop(heap)
        if j in seen:
            continue
        seen.add(j)
        touches += 1
        if j in target:
            hits += 1
        expose(j)

    recall = hits / len(target)
    return (touches if hits >= need else None), edge_inspections, float(recall)


def evaluate_graph(
    graph: LocalGraph,
    groups: Sequence[np.ndarray],
    group_of: np.ndarray,
    target_recall: float = 0.75,
) -> dict:
    """Query every address so added capacity cannot hide as unused padding."""
    n = graph.n_nodes
    touches = []
    logical_work = []
    failures = 0

    for cue in range(n):
        t, edge_checks, _ = query_group_recall(
            graph,
            groups,
            group_of,
            cue,
            target_recall=target_recall,
            max_touches=n,
        )
        if t is None:
            failures += 1
            t = n + 1
        touches.append(float(t))
        logical_work.append(float(t + edge_checks))

    return {
        "mean_touches": float(np.mean(touches)),
        "median_touches": float(np.median(touches)),
        "mean_logical_work": float(np.mean(logical_work)),
        "success_rate": float(1.0 - failures / n),
    }
