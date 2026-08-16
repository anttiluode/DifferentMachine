from __future__ import annotations
import numpy as np
from different_machine.core import (
    MachineSpec,
    EventMachine,
    generate_events,
    receiver_final_contribution_scores,
    replay_candidates,
    topk_mask,
)


def main() -> None:
    T = 10_000
    spec = MachineSpec.random(n_contacts=2048, n_branches=64, n_receivers=2, seed=13)
    events = generate_events(T, spec.n_contacts, event_probability=0.05, seed=17)

    sender = EventMachine(spec)
    full_y, candidates = sender.run(events, T)

    n = len(candidates)
    keep_n = max(1, int(round(0.20 * n)))

    magnitude_scores = np.asarray([abs(e.value) for e in candidates], dtype=np.float64)
    magnitude_mask = topk_mask(magnitude_scores, keep_n)
    magnitude_y = replay_candidates(spec, candidates, T, magnitude_mask)

    print("DifferentMachine Gate 0b — RECEIVER-CONDITIONED ACTIVE FRONTIER")
    print(f"candidate promotions           : {n}")
    print(f"budget                         : {keep_n} ({keep_n/n:.1%})")
    print()

    relation_masks = []

    for r in range(spec.n_receivers):
        # Practical-shaped local score: current event strength times a persistent
        # receiver<->branch relationship weight. It does not look into the future.
        relation_scores = np.asarray(
            [abs(e.value * spec.receiver_weight[r, e.branch]) for e in candidates],
            dtype=np.float64,
        )
        relation_mask = topk_mask(relation_scores, keep_n)
        relation_masks.append(relation_mask)
        relation_y = replay_candidates(spec, candidates, T, relation_mask)

        # Privileged exact final-readout sensitivity: control / ceiling only.
        oracle_scores = receiver_final_contribution_scores(spec, candidates, r, T)
        oracle_mask = topk_mask(oracle_scores, keep_n)
        oracle_y = replay_candidates(spec, candidates, T, oracle_mask)

        mag_err = abs(magnitude_y[r] - full_y[r])
        rel_err = abs(relation_y[r] - full_y[r])
        oracle_err = abs(oracle_y[r] - full_y[r])

        print(f"receiver {r}")
        print(f"  full final readout           : {full_y[r]: .6f}")
        print(f"  magnitude-budget error       : {mag_err:.6e}")
        print(f"  relationship-score error     : {rel_err:.6e}")
        print(f"  final-readout oracle error   : {oracle_err:.6e}")
        print(f"  relation vs magnitude        : {mag_err/max(rel_err,1e-18):.2f}x")
        print()

    if len(relation_masks) >= 2:
        a, b = relation_masks[:2]
        inter = int(np.sum(a & b))
        union = int(np.sum(a | b))
        print("same sender, same budget, different receiver relationship")
        print(f"  selected overlap             : {inter}/{keep_n}")
        print(f"  Jaccard                       : {inter/max(union,1):.4f}")
        print()

    print("RECEIPT ONLY")
    print("The persistent relationship-weight policy is intentionally simple; the exact")
    print("future sensitivity remains an oracle control. Gate 1 must learn/adapt the score")
    print("and pay for routing, state, and reacquisition cost.")


if __name__ == "__main__":
    main()
