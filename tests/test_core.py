import numpy as np

from different_machine.core import (
    ClockedMachine,
    EventMachine,
    MachineSpec,
    generate_events,
    replay_candidates,
    topk_mask,
)


def test_clocked_and_lazy_match():
    T = 1000
    spec = MachineSpec.random(n_contacts=128, n_branches=8, n_receivers=2, seed=1)
    events = generate_events(T, spec.n_contacts, event_probability=0.08, seed=2)
    y_a, c_a = ClockedMachine(spec).run(events, T)
    y_b, c_b = EventMachine(spec).run(events, T)

    assert len(c_a) == len(c_b)
    assert np.max(np.abs(y_a - y_b)) < 1e-12
    assert max(abs(a.value - b.value) for a, b in zip(c_a, c_b)) < 1e-12


def test_event_machine_touches_only_addressed_private_state():
    T = 500
    spec = MachineSpec.random(n_contacts=256, n_branches=16, n_receivers=1, seed=3)
    events = generate_events(T, spec.n_contacts, event_probability=0.10, seed=4)
    machine = EventMachine(spec)
    machine.run(events, T)

    assert machine.contact_touches == len(events)
    assert machine.branch_touches == len(events)


def test_topk_mask_has_exact_budget():
    scores = np.asarray([0.1, 0.4, 0.2, 0.3])
    mask = topk_mask(scores, 2)
    assert int(mask.sum()) == 2
    assert mask[1]
    assert mask[3]


def test_replay_full_mask_matches_full_receiver():
    T = 400
    spec = MachineSpec.random(n_contacts=64, n_branches=8, n_receivers=2, seed=5)
    events = generate_events(T, spec.n_contacts, event_probability=0.12, seed=6)
    machine = EventMachine(spec)
    full_y, candidates = machine.run(events, T)
    mask = np.ones(len(candidates), dtype=bool)
    replay_y = replay_candidates(spec, candidates, T, mask)
    assert np.max(np.abs(full_y - replay_y)) < 1e-12
