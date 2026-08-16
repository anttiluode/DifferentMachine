from __future__ import annotations
import numpy as np

from different_machine.frontier import (
    CascadeSpec,
    value_table,
    learn_receiver_relation,
    predicted_value_table,
    run_budgeted_cascade,
)


def evaluate(rho: float, budget: int, trials: int = 200) -> dict:
    max_depth = 6
    spec = CascadeSpec.random(
        n_branches=64, degree=3, n_receivers=2, rho=rho, seed=101
    )
    true_v = value_table(spec, max_depth)
    learned_k = learn_receiver_relation(
        spec, samples_per_receiver=4096, noise_std=0.08, seed=202
    )
    learned_v = predicted_value_table(spec, learned_k, max_depth)
    generic_v = np.mean(np.abs(learned_v), axis=0)

    rng = np.random.default_rng(303)
    out = {}

    for r in range(spec.n_receivers):
        errors = {"magnitude": [], "generic": [], "receiver": [], "oracle": []}
        for _ in range(trials):
            address = int(rng.integers(0, spec.n_branches))
            value = float(rng.normal())
            full = float(value * true_v[r, max_depth, address])

            policies = {
                "magnitude": lambda i, rem, q: abs(q),
                "generic": lambda i, rem, q: abs(q) * generic_v[rem, i],
                "receiver": lambda i, rem, q: abs(q) * abs(learned_v[r, rem, i]),
                "oracle": lambda i, rem, q: abs(q) * abs(true_v[r, rem, i]),
            }
            for name, fn in policies.items():
                approx, _ = run_budgeted_cascade(
                    spec, address, value, r, max_depth, budget, fn
                )
                errors[name].append(abs(approx - full))

        out[r] = {k: float(np.mean(v)) for k, v in errors.items()}

    rel_rmse = float(np.sqrt(np.mean((learned_k - spec.receiver_weight) ** 2)))
    return {"receivers": out, "relation_rmse": rel_rmse}


def main() -> None:
    print("DifferentMachine Gate 1A — COUPLED ACTIVE-FRONTIER INSTRUMENT")
    print("64 branches, degree 3, depth 6, learned noisy receiver relation")
    print("This is a pre-gate instrument; it does not yet include delta/EGRU/MoE controls.")
    print()

    for rho in (0.15, 0.35, 0.55, 0.75):
        res = evaluate(rho=rho, budget=48, trials=200)
        print(f"rho={rho:.2f}  learned relation RMSE={res['relation_rmse']:.4f}")
        for r, d in res["receivers"].items():
            print(
                f"  receiver {r}: "
                f"mag={d['magnitude']:.3e}  "
                f"generic={d['generic']:.3e}  "
                f"receiver={d['receiver']:.3e}  "
                f"oracle={d['oracle']:.3e}"
            )
        print()

    print("INTERPRETATION GUARDRAIL")
    print("A receiver-conditioned best-first frontier can be learned from local")
    print("interaction in this linear cable toy. This is still system identification")
    print("+ best-first conditional computation. Gate 1 is not positive until the same")
    print("mechanism faces matched delta/event-RNN/MoE controls and router accounting.")


if __name__ == "__main__":
    main()
