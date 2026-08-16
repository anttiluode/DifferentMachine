from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlowEligibilityTrace:
    """Bounded structural eligibility recorded before task outcome arrives."""

    cue: int
    existing_neighbor: int
    existing_delta: float
    candidate_neighbor: int
    candidate_delta: float
    prediction_sign: float


@dataclass
class FlowCreditGraph:
    """Fixed-degree overlay learned from delayed task consequence.

    `proposals[i]` is a bounded physical/candidate neighborhood. It is not a
    learned useful-pair label: the task learner must decide which proposal helps.
    Only the addressed row is aged or changed when credit arrives.
    """

    proposals: np.ndarray
    neighbors: np.ndarray
    strength: np.ndarray
    decay: float = 0.985

    row_updates: int = 0
    slot_ops: int = 0
    proposal_ops: int = 0
    insertions: int = 0
    replacements: int = 0
    reinforcements: int = 0

    @classmethod
    def from_proposals(
        cls,
        proposals: np.ndarray,
        degree: int,
        rng: np.random.Generator,
        decay: float = 0.985,
        initial_strength: float = 0.20,
    ) -> "FlowCreditGraph":
        proposals = np.asarray(proposals, dtype=np.int64)
        if proposals.ndim != 2:
            raise ValueError("proposals must be [n_nodes, proposal_degree]")
        n_nodes, proposal_degree = proposals.shape
        if degree <= 0 or degree > proposal_degree:
            raise ValueError("degree must be in [1, proposal_degree]")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")

        neighbors = np.empty((n_nodes, degree), dtype=np.int64)
        for i in range(n_nodes):
            row = np.unique(proposals[i])
            row = row[row != i]
            if row.size < degree:
                raise ValueError(
                    "each proposal row needs degree distinct non-self nodes"
                )
            neighbors[i] = rng.choice(row, size=degree, replace=False)

        strength = np.full(
            (n_nodes, degree), float(initial_strength), dtype=np.float64
        )
        return cls(
            proposals=proposals.copy(),
            neighbors=neighbors,
            strength=strength,
            decay=float(decay),
        )

    @property
    def n_nodes(self) -> int:
        return int(self.neighbors.shape[0])

    @property
    def degree(self) -> int:
        return int(self.neighbors.shape[1])

    @property
    def proposal_degree(self) -> int:
        return int(self.proposals.shape[1])

    @property
    def persistent_bytes(self) -> int:
        return int(
            self.proposals.nbytes + self.neighbors.nbytes + self.strength.nbytes
        )

    def sample_candidate(self, cue: int, rng: np.random.Generator) -> int:
        """Choose one not-yet-active candidate from a bounded proposal row."""
        cue = int(cue)
        row = self.neighbors[cue]
        candidates = []
        for j in self.proposals[cue]:
            self.proposal_ops += 1
            jj = int(j)
            if jj == cue:
                continue
            self.slot_ops += self.degree
            if not np.any(row == jj):
                candidates.append(jj)

        if not candidates:
            return int(rng.choice(self.proposals[cue]))
        return int(rng.choice(np.asarray(candidates, dtype=np.int64)))

    def apply_credits(
        self,
        cue: int,
        existing_neighbor: int,
        existing_credit: float,
        candidate_neighbor: int,
        candidate_credit: float,
        *,
        lr: float = 0.28,
        insert_threshold: float = 0.012,
    ) -> None:
        """Apply delayed credit by touching one bounded row only."""
        cue = int(cue)
        self.row_updates += 1

        self.strength[cue] *= self.decay
        self.slot_ops += self.degree

        row = self.neighbors[cue]
        weights = self.strength[cue]

        self.slot_ops += self.degree
        hit = np.flatnonzero(row == int(existing_neighbor))
        if hit.size:
            slot = int(hit[0])
            weights[slot] = max(
                0.001, float(weights[slot] + lr * existing_credit)
            )
            self.reinforcements += 1

        if candidate_credit <= insert_threshold:
            return

        self.slot_ops += self.degree
        hit = np.flatnonzero(row == int(candidate_neighbor))
        if hit.size:
            slot = int(hit[0])
            weights[slot] += lr * candidate_credit
            self.reinforcements += 1
            return

        self.slot_ops += self.degree
        free = np.flatnonzero(row < 0)
        if free.size:
            slot = int(free[0])
            self.insertions += 1
        else:
            self.slot_ops += self.degree
            slot = int(np.argmin(weights))
            self.replacements += 1

        row[slot] = int(candidate_neighbor)
        weights[slot] = max(0.05, float(0.15 + lr * candidate_credit))


def consequence_credit(
    reward: float, prediction_sign: float, flow_delta: float
) -> float:
    """Delayed scalar outcome x own output sign x local receiver-flow effect."""
    if reward not in (-1.0, 1.0):
        raise ValueError("reward must be -1 or +1")
    prediction_sign = 1.0 if prediction_sign >= 0.0 else -1.0
    return float(reward * prediction_sign * flow_delta)
