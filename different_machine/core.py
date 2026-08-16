from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class InputEvent:
    t: int
    address: int
    value: float


@dataclass(frozen=True)
class PromotionEvent:
    t: int
    branch: int
    value: float


@dataclass
class MachineSpec:
    contact_to_branch: np.ndarray
    contact_decay: np.ndarray
    contact_gain: np.ndarray
    contact_weight: np.ndarray
    branch_decay: np.ndarray
    branch_gain: np.ndarray
    receiver_decay: np.ndarray
    receiver_weight: np.ndarray

    @property
    def n_contacts(self) -> int:
        return int(self.contact_to_branch.shape[0])

    @property
    def n_branches(self) -> int:
        return int(self.branch_decay.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.receiver_decay.shape[0])

    @classmethod
    def random(
        cls,
        n_contacts: int = 4096,
        n_branches: int = 64,
        n_receivers: int = 2,
        seed: int = 0,
    ) -> "MachineSpec":
        if n_contacts < n_branches:
            raise ValueError("n_contacts must be >= n_branches")
        rng = np.random.default_rng(seed)
        # Compile routing into persistent structure: each contact has one home branch.
        contact_to_branch = np.arange(n_contacts, dtype=np.int64) % n_branches
        rng.shuffle(contact_to_branch)

        # Discrete per-tick decays. Values near 1 create long-lived unfinished state.
        contact_decay = rng.uniform(0.970, 0.9995, size=n_contacts).astype(np.float64)
        contact_gain = rng.uniform(0.6, 1.4, size=n_contacts).astype(np.float64)
        contact_weight = rng.normal(0.0, 0.35, size=n_contacts).astype(np.float64)

        branch_decay = rng.uniform(0.94, 0.995, size=n_branches).astype(np.float64)
        branch_gain = rng.uniform(0.7, 1.3, size=n_branches).astype(np.float64)

        receiver_decay = rng.uniform(0.985, 0.998, size=n_receivers).astype(np.float64)
        receiver_weight = rng.normal(0.0, 0.7, size=(n_receivers, n_branches)).astype(np.float64)

        # Make receiver interests structurally different, not just different random signs.
        if n_receivers >= 2 and n_branches >= 4:
            half = n_branches // 2
            receiver_weight[0, half:] *= 0.20
            receiver_weight[1, :half] *= 0.20

        return cls(
            contact_to_branch=contact_to_branch,
            contact_decay=contact_decay,
            contact_gain=contact_gain,
            contact_weight=contact_weight,
            branch_decay=branch_decay,
            branch_gain=branch_gain,
            receiver_decay=receiver_decay,
            receiver_weight=receiver_weight,
        )


def generate_events(
    T: int,
    n_contacts: int,
    event_probability: float = 0.02,
    seed: int = 1,
) -> List[InputEvent]:
    rng = np.random.default_rng(seed)
    events: List[InputEvent] = []
    for t in range(1, T + 1):
        if rng.random() < event_probability:
            address = int(rng.integers(0, n_contacts))
            value = float(rng.normal())
            events.append(InputEvent(t=t, address=address, value=value))
    return events


PromoteFn = Callable[[PromotionEvent], bool]


class ReceiverBank:
    """Persistent receiver state updated only by promoted events.

    Receiver state decays analytically between touches. A receiver therefore
    exists while quiet without requiring a global recurrent update every tick.
    """

    def __init__(self, spec: MachineSpec):
        self.spec = spec
        self.state = np.zeros(spec.n_receivers, dtype=np.float64)
        self.last_t = np.zeros(spec.n_receivers, dtype=np.int64)
        self.touches = 0

    def _advance_one(self, r: int, t: int) -> None:
        dt = int(t - self.last_t[r])
        if dt < 0:
            raise ValueError("events must be time ordered")
        if dt:
            self.state[r] *= self.spec.receiver_decay[r] ** dt
            self.last_t[r] = t

    def receive(self, event: PromotionEvent) -> None:
        j = event.branch
        for r in range(self.spec.n_receivers):
            self._advance_one(r, event.t)
            self.state[r] += self.spec.receiver_weight[r, j] * event.value
            self.touches += 1

    def read(self, t: int) -> np.ndarray:
        for r in range(self.spec.n_receivers):
            self._advance_one(r, t)
        return self.state.copy()


class EventMachine:
    """Addressed/lazy execution of the persistent machine.

    Only the addressed contact and its home branch are advanced on an input
    event. Quiet local state is stored, not recomputed.
    """

    def __init__(self, spec: MachineSpec):
        self.spec = spec
        self.contact_state = np.zeros(spec.n_contacts, dtype=np.float64)
        self.contact_last_t = np.zeros(spec.n_contacts, dtype=np.int64)
        self.branch_state = np.zeros(spec.n_branches, dtype=np.float64)
        self.branch_last_t = np.zeros(spec.n_branches, dtype=np.int64)
        self.receivers = ReceiverBank(spec)

        self.contact_touches = 0
        self.branch_touches = 0
        self.promotions = 0
        self.candidates = 0

    def process_event(
        self,
        event: InputEvent,
        promote: Optional[PromoteFn] = None,
    ) -> PromotionEvent:
        i = int(event.address)
        if i < 0 or i >= self.spec.n_contacts:
            raise IndexError(i)
        t = int(event.t)

        # Touch exactly one contact.
        dt = int(t - self.contact_last_t[i])
        if dt < 0:
            raise ValueError("events must be time ordered")
        if dt:
            self.contact_state[i] *= self.spec.contact_decay[i] ** dt
        self.contact_last_t[i] = t
        self.contact_state[i] += float(event.value)
        self.contact_touches += 1

        local = self.spec.contact_weight[i] * np.tanh(
            self.spec.contact_gain[i] * self.contact_state[i]
        )

        # The contact's physical address compiles the route to one branch.
        j = int(self.spec.contact_to_branch[i])
        dtb = int(t - self.branch_last_t[j])
        if dtb:
            self.branch_state[j] *= self.spec.branch_decay[j] ** dtb
        self.branch_last_t[j] = t
        self.branch_state[j] += local
        self.branch_touches += 1

        outward = float(self.spec.branch_gain[j] * np.tanh(self.branch_state[j]))
        candidate = PromotionEvent(t=t, branch=j, value=outward)
        self.candidates += 1

        if promote is None or promote(candidate):
            self.receivers.receive(candidate)
            self.promotions += 1
        return candidate

    def run(
        self,
        events: Sequence[InputEvent],
        final_t: int,
        promote: Optional[PromoteFn] = None,
    ) -> Tuple[np.ndarray, List[PromotionEvent]]:
        candidates: List[PromotionEvent] = []
        for event in events:
            candidates.append(self.process_event(event, promote=promote))
        return self.receivers.read(final_t), candidates


class ClockedMachine:
    """Reference execution that explicitly sweeps all represented state each tick."""

    def __init__(self, spec: MachineSpec):
        self.spec = spec
        self.contact_state = np.zeros(spec.n_contacts, dtype=np.float64)
        self.branch_state = np.zeros(spec.n_branches, dtype=np.float64)
        self.receiver_state = np.zeros(spec.n_receivers, dtype=np.float64)
        self.t = 0

        self.contact_touches = 0
        self.branch_touches = 0
        self.receiver_touches = 0
        self.promotions = 0
        self.candidates = 0

    def _tick(self) -> None:
        self.contact_state *= self.spec.contact_decay
        self.branch_state *= self.spec.branch_decay
        self.receiver_state *= self.spec.receiver_decay
        self.contact_touches += self.spec.n_contacts
        self.branch_touches += self.spec.n_branches
        self.receiver_touches += self.spec.n_receivers
        self.t += 1

    def _advance_to(self, t: int) -> None:
        if t < self.t:
            raise ValueError("events must be time ordered")
        while self.t < t:
            self._tick()

    def process_event(
        self,
        event: InputEvent,
        promote: Optional[PromoteFn] = None,
    ) -> PromotionEvent:
        self._advance_to(int(event.t))
        i = int(event.address)
        self.contact_state[i] += float(event.value)
        local = self.spec.contact_weight[i] * np.tanh(
            self.spec.contact_gain[i] * self.contact_state[i]
        )
        j = int(self.spec.contact_to_branch[i])
        self.branch_state[j] += local
        outward = float(self.spec.branch_gain[j] * np.tanh(self.branch_state[j]))
        candidate = PromotionEvent(t=int(event.t), branch=j, value=outward)
        self.candidates += 1

        if promote is None or promote(candidate):
            self.receiver_state += self.spec.receiver_weight[:, j] * outward
            self.promotions += 1
        return candidate

    def run(
        self,
        events: Sequence[InputEvent],
        final_t: int,
        promote: Optional[PromoteFn] = None,
    ) -> Tuple[np.ndarray, List[PromotionEvent]]:
        candidates: List[PromotionEvent] = []
        for event in events:
            candidates.append(self.process_event(event, promote=promote))
        self._advance_to(final_t)
        return self.receiver_state.copy(), candidates


def receiver_final_contribution_scores(
    spec: MachineSpec,
    candidates: Sequence[PromotionEvent],
    receiver: int,
    final_t: int,
) -> np.ndarray:
    """Exact single-event contribution magnitude to one final receiver readout.

    This is an oracle sensitivity used only as an instrument/control. It is not
    claimed as a practical attention algorithm.
    """
    decay = float(spec.receiver_decay[receiver])
    scores = np.empty(len(candidates), dtype=np.float64)
    for k, e in enumerate(candidates):
        scores[k] = abs(
            spec.receiver_weight[receiver, e.branch]
            * e.value
            * (decay ** (final_t - e.t))
        )
    return scores


def replay_candidates(
    spec: MachineSpec,
    candidates: Sequence[PromotionEvent],
    final_t: int,
    keep_mask: np.ndarray,
) -> np.ndarray:
    bank = ReceiverBank(spec)
    for keep, e in zip(keep_mask, candidates):
        if bool(keep):
            bank.receive(e)
    return bank.read(final_t)


def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    n = int(scores.shape[0])
    k = max(0, min(int(k), n))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    if k == n:
        mask[:] = True
        return mask
    idx = np.argpartition(scores, -k)[-k:]
    mask[idx] = True
    return mask
