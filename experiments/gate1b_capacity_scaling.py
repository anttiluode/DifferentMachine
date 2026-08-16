from __future__ import annotations

import heapq
import math
import numpy as np

from different_machine.frontier import (
    CascadeSpec,
    value_table,
    learn_receiver_relation,
    predicted_value_table,
)


CAPACITIES = (32, 64, 128, 256, 512)
BUDGETS = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
MAX_DEPTH = 5
RHO = 0.55
TARGET_NRMSE = 0.01
REPEATS_PER_ADDRESS = 2
SAMPLES_PER_BRANCH = 24


def _cascade_trace(
    spec: CascadeSpec,
    address: int,
    value: float,
    receiver: int,
    max_depth: int,
    max_budget: int,
    score_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return prefix receiver outputs and cumulative scorer calls.

    One branch expansion counts as one branch touch. Every candidate pushed onto
    the priority queue requires one receiver-conditioned score lookup and counts
    as one scorer call. Heap/runtime overhead is intentionally *not* folded into
    this logical work metric; Gate 1 hardware accounting remains separate.
    """
    heap: list[tuple[float, int, int, int, float]] = []
    counter = 0
    scorer_calls = 0

    def push(i: int, rem: int, q: float) -> None:
        nonlocal counter, scorer_calls
        score = abs(q) * abs(float(score_table[rem, i]))
        heapq.heappush(heap, (-score, counter, i, rem, q))
        counter += 1
        scorer_calls += 1

    push(int(address), int(max_depth), float(value))
    y = 0.0
    prefix_y: list[float] = []
    prefix_calls: list[int] = []

    while heap and len(prefix_y) < max_budget:
        _, _, i, rem, q = heapq.heappop(heap)
        y += float(spec.receiver_weight[receiver, i]) * q

        if rem > 0:
            for j, w in zip(spec.neighbors[i], spec.edge_weight[i]):
                push(int(j), rem - 1, q * spec.rho * float(w))

        prefix_y.append(y)
        prefix_calls.append(scorer_calls)

    return np.asarray(prefix_y), np.asarray(prefix_calls)


def _degree_for(regime: str, n: int) -> int:
    if regime == "bounded":
        return 3
    if regime == "sqrt":
        return max(3, int(round(math.sqrt(n))))
    if regime == "mixing":
        return max(3, n // 8)
    raise ValueError(regime)


def evaluate_capacity(regime: str, n: int) -> dict:
    degree = _degree_for(regime, n)
    spec = CascadeSpec.random(
        n_branches=n,
        degree=degree,
        n_receivers=2,
        rho=RHO,
        seed=1000 + n + degree,
    )

    true_v = value_table(spec, MAX_DEPTH)
    learned_k = learn_receiver_relation(
        spec,
        samples_per_receiver=SAMPLES_PER_BRANCH * n,
        noise_std=0.08,
        seed=2000 + n + degree,
    )
    learned_v = predicted_value_table(spec, learned_k, MAX_DEPTH)

    # Every branch is used exactly REPEATS_PER_ADDRESS times as the originating
    # event address. Increasing N therefore increases genuinely addressable
    # sender capacity rather than appending never-used padding.
    addresses = np.tile(np.arange(n, dtype=np.int64), REPEATS_PER_ADDRESS)
    rng = np.random.default_rng(3000 + n + degree)
    rng.shuffle(addresses)
    values = rng.normal(size=len(addresses))
    receivers = rng.integers(0, spec.n_receivers, size=len(addresses))

    full = values * true_v[receivers, MAX_DEPTH, addresses]
    full_rms = float(np.sqrt(np.mean(full**2))) + 1e-12

    sse = {b: 0.0 for b in BUDGETS}
    scorer_sum = {b: 0.0 for b in BUDGETS}
    touch_sum = {b: 0.0 for b in BUDGETS}
    max_budget = max(BUDGETS)

    for address, value, receiver, full_y in zip(
        addresses, values, receivers, full
    ):
        r = int(receiver)
        prefix_y, prefix_calls = _cascade_trace(
            spec,
            int(address),
            float(value),
            r,
            MAX_DEPTH,
            max_budget,
            learned_v[r],
        )

        for budget in BUDGETS:
            idx = min(budget, len(prefix_y)) - 1
            err = float(prefix_y[idx] - full_y)
            sse[budget] += err * err
            scorer_sum[budget] += float(prefix_calls[idx])
            touch_sum[budget] += float(idx + 1)

    count = float(len(addresses))
    nrmse = {
        b: math.sqrt(sse[b] / count) / full_rms
        for b in BUDGETS
    }
    required_budget = next(
        (b for b in BUDGETS if nrmse[b] <= TARGET_NRMSE),
        max(BUDGETS),
    )

    mean_touches = touch_sum[required_budget] / count
    mean_scorer = scorer_sum[required_budget] / count

    # A clocked/global propagation would at minimum carry N branch states over
    # each of MAX_DEPTH+1 receiver-contributing levels. This is a logical state-
    # touch reference, not a FLOP or wall-clock claim.
    global_clock_touches = n * (MAX_DEPTH + 1)

    return {
        "n": n,
        "degree": degree,
        "budget": required_budget,
        "nrmse": nrmse[required_budget],
        "branch_touches": mean_touches,
        "scorer_calls": mean_scorer,
        "logical_work": mean_touches + mean_scorer,
        "global_clock_touches": float(global_clock_touches),
        "relation_rmse": float(
            np.sqrt(np.mean((learned_k - spec.receiver_weight) ** 2))
        ),
        "relation_bytes": int(learned_k.nbytes),
        "acquisition_samples": int(
            spec.n_receivers * SAMPLES_PER_BRANCH * n
        ),
        "address_coverage": float(len(np.unique(addresses)) / n),
    }


def _fit_alpha(rows: list[dict], key: str) -> float:
    x = np.log(np.asarray([row["n"] for row in rows], dtype=np.float64))
    y = np.log(np.asarray([row[key] for row in rows], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    print("DifferentMachine Gate 1B — CAPACITY VS PER-EVENT WORK")
    print(
        f"target NRMSE={TARGET_NRMSE:.3f}, rho={RHO:.2f}, depth={MAX_DEPTH}, "
        f"every address used {REPEATS_PER_ADDRESS}x"
    )
    print()

    all_rows: dict[str, list[dict]] = {}

    for regime in ("bounded", "sqrt", "mixing"):
        rows = []
        print(f"TOPOLOGY: {regime}")
        print(
            "N     degree  Kreq  NRMSE      branch  scorer   logical   global   relKB"
        )
        for n in CAPACITIES:
            row = evaluate_capacity(regime, n)
            rows.append(row)
            print(
                f"{row['n']:4d}  {row['degree']:6d}  {row['budget']:4d}  "
                f"{row['nrmse']:.4e}  {row['branch_touches']:7.1f}  "
                f"{row['scorer_calls']:7.1f}  {row['logical_work']:8.1f}  "
                f"{row['global_clock_touches']:7.0f}  "
                f"{row['relation_bytes']/1024:6.1f}"
            )
            assert row["address_coverage"] == 1.0

        alpha_work = _fit_alpha(rows, "logical_work")
        alpha_global = _fit_alpha(rows, "global_clock_touches")
        all_rows[regime] = rows
        print(
            f"fit: logical work ~ N^{alpha_work:.3f}    "
            f"global clock reference ~ N^{alpha_global:.3f}"
        )
        print()

    alpha_bounded = _fit_alpha(all_rows["bounded"], "logical_work")
    alpha_sqrt = _fit_alpha(all_rows["sqrt"], "logical_work")
    alpha_mixing = _fit_alpha(all_rows["mixing"], "logical_work")

    print("INTERPRETATION")
    print(
        "This receipt asks a scaling question, not a novelty question. With bounded "
        "causal degree, added addressable capacity can remain mostly dormant per "
        "event and required logical work can stay approximately flat. As fanout "
        "densifies with N, merely discovering/scoring the causal frontier becomes "
        "more expensive and the scaling approaches linear."
    )
    print()
    print(
        f"observed alpha: bounded={alpha_bounded:.3f}, "
        f"sqrt-degree={alpha_sqrt:.3f}, mixing={alpha_mixing:.3f}"
    )
    print(
        "Relationship memory and acquisition samples still grow with N. The point is "
        "resource separation: represented memory/capacity may scale differently from "
        "per-event executed computation."
    )
    print()
    print("GUARDRAIL")
    print(
        "Bounded-degree local computation and sparse/event execution are established "
        "ideas. Gate 1B does not establish a new architecture result. DifferentMachine "
        "still has to beat matched delta/event-RNN/MoE/generic-router controls and "
        "actual hardware runtime in the full Gate 1 benchmark."
    )


if __name__ == "__main__":
    main()
