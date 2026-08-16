from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class OnlineLocalGraph:
    """Fixed-degree receiver-local adjacency updated only where experience lands.

    Each directed row stores `degree` neighbor ids plus one strength per slot.
    Updating pair (i, j) touches only rows i and j. No global pair table or scan
    exists. Strengths decay only when their row is locally touched, allowing old
    associations to lose priority after a regime change.

    This is a deliberately simple bounded-memory plasticity heuristic, not a
    novelty claim.
    """

    neighbors: np.ndarray
    strength: np.ndarray
    decay: float = 0.97

    row_updates: int = 0
    slot_ops: int = 0
    reinforcements: int = 0
    insertions: int = 0
    replacements: int = 0

    @classmethod
    def empty(cls, n_nodes: int, degree: int = 4, decay: float = 0.97) -> "OnlineLocalGraph":
        if n_nodes <= 0 or degree <= 0:
            raise ValueError("n_nodes and degree must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        return cls(
            neighbors=np.full((n_nodes, degree), -1, dtype=np.int64),
            strength=np.zeros((n_nodes, degree), dtype=np.float64),
            decay=float(decay),
        )

    @property
    def n_nodes(self) -> int:
        return int(self.neighbors.shape[0])

    @property
    def degree(self) -> int:
        return int(self.neighbors.shape[1])

    @property
    def persistent_bytes(self) -> int:
        return int(self.neighbors.nbytes + self.strength.nbytes)

    def _update_directed(self, i: int, j: int) -> None:
        """Update one bounded adjacency row in O(degree) work."""
        if i == j:
            return
        d = self.degree
        self.row_updates += 1

        # Local forgetting: only this touched row is aged.
        self.strength[i] *= self.decay
        self.slot_ops += d

        row = self.neighbors[i]
        weights = self.strength[i]

        # Full bounded row scan for an existing edge.
        self.slot_ops += d
        hit = np.flatnonzero(row == j)
        if hit.size:
            weights[int(hit[0])] += 1.0
            self.reinforcements += 1
            return

        # Full bounded row scan for an unused slot.
        self.slot_ops += d
        free = np.flatnonzero(row < 0)
        if free.size:
            slot = int(free[0])
            row[slot] = int(j)
            weights[slot] = 1.0
            self.insertions += 1
            return

        # Full bounded row scan to evict the weakest surviving relation.
        self.slot_ops += d
        slot = int(np.argmin(weights))
        row[slot] = int(j)
        weights[slot] = 1.0
        self.replacements += 1

    def update_pair(self, i: int, j: int) -> None:
        """Hebbian-style pair event: touch only the two addressed endpoint rows."""
        i = int(i)
        j = int(j)
        if i < 0 or i >= self.n_nodes or j < 0 or j >= self.n_nodes:
            raise IndexError("pair endpoint out of range")
        if i == j:
            return
        self._update_directed(i, j)
        self._update_directed(j, i)


@dataclass
class RecentLocalGraph:
    """Same-memory recency-only control.

    It keeps the most recently observed distinct neighbors rather than repeated
    co-use strength. The storage shape is identical to `OnlineLocalGraph`.
    """

    neighbors: np.ndarray
    strength: np.ndarray
    clock: int = 0
    row_updates: int = 0
    slot_ops: int = 0

    @classmethod
    def empty(cls, n_nodes: int, degree: int = 4) -> "RecentLocalGraph":
        return cls(
            neighbors=np.full((n_nodes, degree), -1, dtype=np.int64),
            strength=np.zeros((n_nodes, degree), dtype=np.float64),
        )

    @property
    def n_nodes(self) -> int:
        return int(self.neighbors.shape[0])

    @property
    def degree(self) -> int:
        return int(self.neighbors.shape[1])

    @property
    def persistent_bytes(self) -> int:
        return int(self.neighbors.nbytes + self.strength.nbytes)

    def _update_directed(self, i: int, j: int) -> None:
        d = self.degree
        self.clock += 1
        self.row_updates += 1
        row = self.neighbors[i]
        stamp = self.strength[i]

        self.slot_ops += d
        hit = np.flatnonzero(row == j)
        if hit.size:
            stamp[int(hit[0])] = float(self.clock)
            return

        self.slot_ops += d
        free = np.flatnonzero(row < 0)
        if free.size:
            slot = int(free[0])
        else:
            self.slot_ops += d
            slot = int(np.argmin(stamp))
        row[slot] = int(j)
        stamp[slot] = float(self.clock)

    def update_pair(self, i: int, j: int) -> None:
        if i == j:
            return
        self._update_directed(int(i), int(j))
        self._update_directed(int(j), int(i))


def sample_receiver_couse_pair(
    groups_all: Sequence[Sequence[np.ndarray]],
    group_of_all: Sequence[np.ndarray],
    receiver: int,
    rng: np.random.Generator,
    signal_probability: float = 0.80,
) -> Tuple[int, int]:
    """Generate one environment co-use event without exposing latent labels.

    The learner receives only `(receiver, i, j)`. Latent partitions are used by
    the synthetic world/evaluator to decide which pairs tend to co-occur.
    """
    if not 0.0 <= signal_probability <= 1.0:
        raise ValueError("signal_probability must be in [0, 1]")

    r = int(receiver)
    n_receivers = len(groups_all)
    n_nodes = len(group_of_all[r])
    other = (r + 1) % n_receivers if n_receivers > 1 else r

    i = int(rng.integers(0, n_nodes))
    source_r = r if rng.random() < signal_probability else other
    members = groups_all[source_r][int(group_of_all[source_r][i])]
    candidates = members[members != i]
    j = int(rng.choice(candidates))
    return i, j
