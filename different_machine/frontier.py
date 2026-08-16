from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Tuple
import numpy as np


@dataclass
class CascadeSpec:
    neighbors: np.ndarray
    edge_weight: np.ndarray
    receiver_weight: np.ndarray
    rho: float = 0.45

    @property
    def n_branches(self) -> int:
        return int(self.neighbors.shape[0])

    @property
    def degree(self) -> int:
        return int(self.neighbors.shape[1])

    @property
    def n_receivers(self) -> int:
        return int(self.receiver_weight.shape[0])

    @classmethod
    def random(
        cls,
        n_branches: int = 64,
        degree: int = 3,
        n_receivers: int = 2,
        rho: float = 0.45,
        seed: int = 0,
    ) -> "CascadeSpec":
        rng = np.random.default_rng(seed)
        neighbors = np.empty((n_branches, degree), dtype=np.int64)
        edge_weight = np.empty((n_branches, degree), dtype=np.float64)

        for i in range(n_branches):
            pool = np.delete(np.arange(n_branches), i)
            nbr = rng.choice(pool, size=degree, replace=False)
            w = rng.normal(0.0, 1.0, size=degree)
            w /= max(np.sum(np.abs(w)), 1e-12)
            neighbors[i] = nbr
            edge_weight[i] = w

        receiver_weight = rng.normal(0.0, 1.0, size=(n_receivers, n_branches))
        if n_receivers >= 2:
            half = n_branches // 2
            receiver_weight[0, half:] *= 0.15
            receiver_weight[1, :half] *= 0.15

        return cls(
            neighbors=neighbors,
            edge_weight=edge_weight,
            receiver_weight=receiver_weight.astype(np.float64),
            rho=float(rho),
        )


def value_table(spec: CascadeSpec, max_depth: int) -> np.ndarray:
    """Exact receiver consequence of fully expanding a unit event.

    V[r, d, i] is the final receiver contribution of a unit event currently at
    branch i when d more propagation levels are available.
    """
    R, B = spec.receiver_weight.shape
    V = np.zeros((R, max_depth + 1, B), dtype=np.float64)
    V[:, 0, :] = spec.receiver_weight

    for d in range(1, max_depth + 1):
        V[:, d, :] = spec.receiver_weight
        for i in range(B):
            nbr = spec.neighbors[i]
            w = spec.edge_weight[i]
            V[:, d, i] += spec.rho * np.sum(V[:, d - 1, nbr] * w[None, :], axis=1)
    return V


def learn_receiver_relation(
    spec: CascadeSpec,
    samples_per_receiver: int = 4096,
    noise_std: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Learn direct branch->receiver relation from noisy local interaction.

    Training samples are random addressed branch pulses q. The receiver observes
    its immediate consequence y = k[r, branch] * q + noise. This is deliberately
    simple system identification; it is relationship acquisition, not a novelty
    claim.
    """
    rng = np.random.default_rng(seed)
    R, B = spec.receiver_weight.shape
    num = np.zeros((R, B), dtype=np.float64)
    den = np.zeros((R, B), dtype=np.float64)

    for r in range(R):
        for _ in range(samples_per_receiver):
            i = int(rng.integers(0, B))
            q = float(rng.normal())
            y = float(spec.receiver_weight[r, i] * q + rng.normal(scale=noise_std))
            num[r, i] += q * y
            den[r, i] += q * q

    return num / np.maximum(den, 1e-9)


def predicted_value_table(
    spec: CascadeSpec,
    learned_relation: np.ndarray,
    max_depth: int,
) -> np.ndarray:
    """Propagate learned receiver relation through known local cable structure."""
    R, B = learned_relation.shape
    V = np.zeros((R, max_depth + 1, B), dtype=np.float64)
    V[:, 0, :] = learned_relation
    for d in range(1, max_depth + 1):
        V[:, d, :] = learned_relation
        for i in range(B):
            nbr = spec.neighbors[i]
            w = spec.edge_weight[i]
            V[:, d, i] += spec.rho * np.sum(V[:, d - 1, nbr] * w[None, :], axis=1)
    return V


def run_budgeted_cascade(
    spec: CascadeSpec,
    address: int,
    value: float,
    receiver: int,
    max_depth: int,
    budget: int,
    score_fn: Callable[[int, int, float], float],
) -> Tuple[float, int]:
    """Best-first expansion of an addressed event through cable space.

    A frontier item is (branch, remaining_depth, event_value). Processing one
    item is one branch-state touch. Children do not exist computationally until
    their parent is expanded.
    """
    counter = 0
    heap = []

    def push(i: int, rem: int, q: float) -> None:
        nonlocal counter
        score = float(score_fn(i, rem, q))
        heapq.heappush(heap, (-score, counter, i, rem, q))
        counter += 1

    push(int(address), int(max_depth), float(value))
    y = 0.0
    touches = 0

    while heap and touches < budget:
        _, _, i, rem, q = heapq.heappop(heap)
        y += spec.receiver_weight[receiver, i] * q
        touches += 1

        if rem > 0:
            for j, w in zip(spec.neighbors[i], spec.edge_weight[i]):
                child_q = q * spec.rho * float(w)
                push(int(j), rem - 1, child_q)

    return float(y), touches
