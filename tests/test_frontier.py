import numpy as np

from different_machine.frontier import (
    CascadeSpec,
    value_table,
    run_budgeted_cascade,
    learn_receiver_relation,
)


def test_full_cascade_matches_value_table():
    spec = CascadeSpec.random(
        n_branches=16, degree=2, n_receivers=2, rho=0.4, seed=21
    )
    depth = 4
    V = value_table(spec, depth)
    address = 3
    value = -0.7

    for r in range(spec.n_receivers):
        y, touches = run_budgeted_cascade(
            spec,
            address=address,
            value=value,
            receiver=r,
            max_depth=depth,
            budget=10_000,
            score_fn=lambda i, rem, q: abs(q),
        )
        expected = value * V[r, depth, address]
        assert abs(y - expected) < 1e-12
        assert touches == sum(spec.degree ** d for d in range(depth + 1))


def test_relation_learning_is_finite_and_shaped():
    spec = CascadeSpec.random(
        n_branches=12, degree=2, n_receivers=2, rho=0.3, seed=22
    )
    k = learn_receiver_relation(
        spec, samples_per_receiver=1000, noise_std=0.05, seed=23
    )
    assert k.shape == spec.receiver_weight.shape
    assert np.isfinite(k).all()
