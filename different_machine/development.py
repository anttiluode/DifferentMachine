from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class DevelopmentalGenome:
    """Small shared rule that grows a receiver-local graph during a lifetime.

    The genome does not contain node ids or per-edge parameters. Every node uses
    the same ten raw genes, decoded to bounded coefficients. A phenotype can
    therefore be discarded while the developmental rule remains constant-size.
    """

    raw: np.ndarray

    N_GENES = 10

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw, dtype=np.float64)
        if raw.shape != (self.N_GENES,):
            raise ValueError(f"raw genome must have shape ({self.N_GENES},)")
        object.__setattr__(self, "raw", raw.copy())

    @property
    def genome_bytes(self) -> int:
        return int(self.raw.nbytes)

    @property
    def distance_weight(self) -> float:
        return -3.0 * _sigmoid(float(self.raw[0]))

    @property
    def activity_weight(self) -> float:
        return 4.0 * math.tanh(float(self.raw[1]))

    @property
    def phase_weight(self) -> float:
        return 2.0 * math.tanh(float(self.raw[2]))

    @property
    def growth_bias(self) -> float:
        return 2.0 * math.tanh(float(self.raw[3]))

    @property
    def initial_strength(self) -> float:
        return 0.08 + 0.92 * _sigmoid(float(self.raw[4]))

    @property
    def local_decay(self) -> float:
        return 0.90 + 0.099 * _sigmoid(float(self.raw[5]))

    @property
    def credit_rate(self) -> float:
        return 0.03 + 0.70 * _sigmoid(float(self.raw[6]))

    @property
    def prune_threshold(self) -> float:
        return 0.01 + 0.30 * _sigmoid(float(self.raw[7]))

    @property
    def activity_decay(self) -> float:
        return 0.70 + 0.29 * _sigmoid(float(self.raw[8]))

    @property
    def phase_drive(self) -> float:
        return 0.55 * math.tanh(float(self.raw[9]))

    def decoded(self) -> dict[str, float]:
        return {
            "distance_weight": self.distance_weight,
            "activity_weight": self.activity_weight,
            "phase_weight": self.phase_weight,
            "growth_bias": self.growth_bias,
            "initial_strength": self.initial_strength,
            "local_decay": self.local_decay,
            "credit_rate": self.credit_rate,
            "prune_threshold": self.prune_threshold,
            "activity_decay": self.activity_decay,
            "phase_drive": self.phase_drive,
        }


@dataclass(frozen=True)
class EdgeEligibility:
    """Residue left by ordinary signal transport across one active edge."""

    source: int
    target: int
    contribution: float


class SpatialHash:
    """Geometry-derived local encounter index.

    There is no hand-prepared proposal matrix. Candidate relations are generated
    from node positions by examining only the 3x3 neighborhood of the cue's
    spatial cell. At constant spatial density and fixed radius, expected
    discovery work is independent of total represented capacity apart from
    boundary/distribution effects.
    """

    def __init__(self, positions: np.ndarray, radius: float = 2.15) -> None:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape [n_nodes, 2]")
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        self.positions = positions.copy()
        self.radius = float(radius)
        self.cell_size = float(radius)
        self._buckets: dict[tuple[int, int], list[int]] = {}
        for i, (x, y) in enumerate(self.positions):
            key = self._cell(float(x), float(y))
            self._buckets.setdefault(key, []).append(int(i))

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor(x / self.cell_size)),
            int(math.floor(y / self.cell_size)),
        )

    @property
    def persistent_bytes_lower_bound(self) -> int:
        return int(self.positions.nbytes)

    def nearby(self, node: int) -> tuple[list[tuple[int, float]], int]:
        node = int(node)
        if node < 0 or node >= len(self.positions):
            raise IndexError("node out of range")
        x, y = self.positions[node]
        cx, cy = self._cell(float(x), float(y))
        radius2 = self.radius * self.radius
        result: list[tuple[int, float]] = []
        inspections = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in self._buckets.get((cx + dx, cy + dy), ()):
                    inspections += 1
                    if other == node:
                        continue
                    delta = self.positions[node] - self.positions[other]
                    distance2 = float(np.dot(delta, delta))
                    if distance2 <= radius2:
                        result.append((int(other), math.sqrt(distance2)))
        return result, inspections


class DevelopmentalGraph:
    """Fixed-degree phenotype grown by one shared genome.

    Edge candidacy comes from `SpatialHash`, not from a stored useful-neighbor
    list. Normal query transport can leave `EdgeEligibility` records; delayed
    task reward later changes only the rows/slots that actually carried signal.
    """

    def __init__(
        self,
        spatial: SpatialHash,
        genome: DevelopmentalGenome,
        degree: int = 4,
        *,
        dynamic_phase: bool = True,
    ) -> None:
        if degree <= 0:
            raise ValueError("degree must be positive")
        self.spatial = spatial
        self.genome = genome
        self.degree = int(degree)
        self.dynamic_phase = bool(dynamic_phase)
        n_nodes = len(spatial.positions)
        self.neighbors = np.full((n_nodes, degree), -1, dtype=np.int64)
        self.strength = np.zeros((n_nodes, degree), dtype=np.float64)
        self.activity = np.zeros(n_nodes, dtype=np.float64)
        self.phase = np.zeros(n_nodes, dtype=np.float64)

        self.candidate_inspections = 0
        self.growths = 0
        self.prunes = 0
        self.credit_updates = 0
        self.row_updates = 0

    @property
    def n_nodes(self) -> int:
        return int(self.neighbors.shape[0])

    @property
    def persistent_bytes(self) -> int:
        return int(
            self.neighbors.nbytes
            + self.strength.nbytes
            + self.activity.nbytes
            + self.phase.nbytes
        )

    @property
    def active_edges(self) -> int:
        return int(np.count_nonzero(self.neighbors >= 0))

    def update_local_state(self, node: int, signal: float) -> None:
        node = int(node)
        decay = self.genome.activity_decay
        self.activity[node] = (
            decay * self.activity[node] + (1.0 - decay) * math.tanh(float(signal))
        )
        if self.dynamic_phase:
            self.phase[node] = (
                self.phase[node]
                + 0.23
                + self.genome.phase_drive * math.tanh(float(signal))
            ) % (2.0 * math.pi)

    def _candidate_score(self, source: int, target: int, distance: float) -> float:
        normalized_distance = float(distance / self.spatial.radius)
        activity_similarity = float(self.activity[source] * self.activity[target])
        phase_similarity = (
            math.cos(float(self.phase[source] - self.phase[target]))
            if self.dynamic_phase
            else 1.0
        )
        return float(
            self.genome.growth_bias
            + self.genome.distance_weight * normalized_distance
            + self.genome.activity_weight * activity_similarity
            + self.genome.phase_weight * phase_similarity
        )

    def grow_once(self, source: int) -> None:
        """Attempt one geometry-local edge birth from an addressed node."""
        source = int(source)
        candidates, inspections = self.spatial.nearby(source)
        self.candidate_inspections += int(inspections)
        if not candidates:
            return

        active = {int(j) for j in self.neighbors[source] if j >= 0}
        best_score = -math.inf
        best_target = -1
        for target, distance in candidates:
            if target in active:
                continue
            score = self._candidate_score(source, int(target), float(distance))
            if score > best_score:
                best_score = score
                best_target = int(target)

        if best_target < 0 or best_score <= 0.0:
            return

        free = np.flatnonzero(self.neighbors[source] < 0)
        if free.size:
            slot = int(free[0])
        else:
            slot = int(np.argmin(self.strength[source]))
            if (
                self.strength[source, slot] > self.genome.initial_strength
                and best_score < 0.75
            ):
                return
            self.prunes += 1

        self.neighbors[source, slot] = best_target
        self.strength[source, slot] = self.genome.initial_strength
        self.growths += 1

    def apply_delayed_credit(
        self, traces: Iterable[EdgeEligibility], reward: float
    ) -> None:
        """Reward-modulate only edges that actually carried the earlier signal."""
        if reward not in (-1.0, 1.0):
            raise ValueError("reward must be -1 or +1")

        touched_rows: set[int] = set()
        for trace in traces:
            source = int(trace.source)
            target = int(trace.target)
            hit = np.flatnonzero(self.neighbors[source] == target)
            if not hit.size:
                continue
            slot = int(hit[0])
            self.strength[source, slot] = max(
                0.0,
                float(
                    self.strength[source, slot]
                    + self.genome.credit_rate * reward * float(trace.contribution)
                ),
            )
            self.credit_updates += 1
            touched_rows.add(source)

        for source in touched_rows:
            self.row_updates += 1
            mask = self.neighbors[source] >= 0
            self.strength[source, mask] *= self.genome.local_decay
            dead = np.flatnonzero(
                mask & (self.strength[source] < self.genome.prune_threshold)
            )
            for slot in dead:
                self.neighbors[source, slot] = -1
                self.strength[source, slot] = 0.0
                self.prunes += 1
