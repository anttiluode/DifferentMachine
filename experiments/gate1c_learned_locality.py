from __future__ import annotations

import numpy as np

from different_machine.plasticity import (
    collect_receiver_couse_counts,
    compile_topk_graph,
    evaluate_graph,
    random_graph,
)


CAPACITIES = (64, 128, 256, 512)
GROUP_SIZE = 8
DEGREE = 4
N_RECEIVERS = 2
SAMPLES_PER_NODE = 60
SIGNAL_PROBABILITY = 0.80
TARGET_RECALL = 0.75


def fit_alpha(ns, work) -> float:
    return float(np.polyfit(np.log(np.asarray(ns, float)), np.log(np.asarray(work, float)), 1)[0])


def run_conflicting() -> dict:
    rows = []
    for n in CAPACITIES:
        receiver_counts, pooled_counts, groups, group_of, events = collect_receiver_couse_counts(
            n_nodes=n,
            group_size=GROUP_SIZE,
            n_receivers=N_RECEIVERS,
            samples_per_node=SAMPLES_PER_NODE,
            signal_probability=SIGNAL_PROBABILITY,
            seed=1000 + n,
            aligned=False,
        )

        pooled = compile_topk_graph(pooled_counts, DEGREE)
        receiver_graphs = [compile_topk_graph(receiver_counts[r], DEGREE) for r in range(N_RECEIVERS)]
        rnd = random_graph(n, DEGREE, seed=2000 + n)

        pooled_eval = [evaluate_graph(pooled, groups[r], group_of[r], TARGET_RECALL) for r in range(N_RECEIVERS)]
        receiver_eval = [
            evaluate_graph(receiver_graphs[r], groups[r], group_of[r], TARGET_RECALL)
            for r in range(N_RECEIVERS)
        ]
        random_eval = [evaluate_graph(rnd, groups[r], group_of[r], TARGET_RECALL) for r in range(N_RECEIVERS)]

        rows.append(
            {
                "n": n,
                "events": events,
                "pooled": pooled_eval,
                "receiver": receiver_eval,
                "random": random_eval,
                "pooled_bytes": pooled.persistent_bytes,
                "receiver_bytes": sum(g.persistent_bytes for g in receiver_graphs),
            }
        )
    return {"rows": rows}


def run_aligned_control() -> dict:
    rows = []
    for n in CAPACITIES:
        receiver_counts, pooled_counts, groups, group_of, events = collect_receiver_couse_counts(
            n_nodes=n,
            group_size=GROUP_SIZE,
            n_receivers=N_RECEIVERS,
            samples_per_node=SAMPLES_PER_NODE,
            signal_probability=SIGNAL_PROBABILITY,
            seed=3000 + n,
            aligned=True,
        )
        pooled = compile_topk_graph(pooled_counts, DEGREE)
        e = evaluate_graph(pooled, groups[0], group_of[0], TARGET_RECALL)
        rows.append({"n": n, "events": events, "pooled": e})
    return {"rows": rows}


def main() -> None:
    print("DifferentMachine Gate 1C — CAN USEFUL LOCALITY BE LEARNED?")
    print(
        f"group={GROUP_SIZE}, local degree={DEGREE}, target recall={TARGET_RECALL:.0%}, "
        f"{SAMPLES_PER_NODE} co-use events/node/receiver"
    )
    print("Learner sees receiver-labelled co-use pairs, never latent group labels.")
    print("The compiled graph has fixed degree; queries may inspect only locally exposed edges.")
    print()

    conflict = run_conflicting()
    print("CONFLICTING RECEIVER LOCALITIES")
    print("N    pooled_work   receiver_work   random_work   pooled_KB   receiver_KB")

    pooled_work = []
    receiver_work = []
    random_work = []
    for row in conflict["rows"]:
        p = float(np.mean([x["mean_logical_work"] for x in row["pooled"]]))
        r = float(np.mean([x["mean_logical_work"] for x in row["receiver"]]))
        q = float(np.mean([x["mean_logical_work"] for x in row["random"]]))
        pooled_work.append(p)
        receiver_work.append(r)
        random_work.append(q)
        print(
            f"{row['n']:4d}  {p:11.1f}   {r:13.1f}   {q:11.1f}   "
            f"{row['pooled_bytes']/1024:8.1f}   {row['receiver_bytes']/1024:11.1f}"
        )
        for recv, e in enumerate(row["receiver"]):
            if e["success_rate"] < 0.999:
                raise AssertionError(f"receiver-conditioned graph failed at N={row['n']} r={recv}")

    alpha_pooled = fit_alpha(CAPACITIES, pooled_work)
    alpha_receiver = fit_alpha(CAPACITIES, receiver_work)
    alpha_random = fit_alpha(CAPACITIES, random_work)
    print(
        f"fit logical work: pooled ~ N^{alpha_pooled:.3f}, "
        f"receiver-conditioned ~ N^{alpha_receiver:.3f}, random ~ N^{alpha_random:.3f}"
    )
    print()

    print("PER-RECEIVER TOUCH DETAIL")
    for row in conflict["rows"]:
        r0 = row["receiver"][0]
        r1 = row["receiver"][1]
        p0 = row["pooled"][0]
        p1 = row["pooled"][1]
        print(
            f"N={row['n']:4d}  pooled touches r0/r1={p0['mean_touches']:.1f}/{p1['mean_touches']:.1f}  "
            f"receiver touches r0/r1={r0['mean_touches']:.1f}/{r1['mean_touches']:.1f}"
        )
    print()

    aligned = run_aligned_control()
    aligned_work = [row["pooled"]["mean_logical_work"] for row in aligned["rows"]]
    alpha_aligned = fit_alpha(CAPACITIES, aligned_work)
    print("ALIGNED-RECEIVER CONTROL")
    print("When both receivers want the same locality, one pooled graph should suffice.")
    for row in aligned["rows"]:
        e = row["pooled"]
        print(
            f"N={row['n']:4d}  touches={e['mean_touches']:.1f}  "
            f"logical_work={e['mean_logical_work']:.1f}  success={e['success_rate']:.3f}"
        )
    print(f"aligned pooled fit: work ~ N^{alpha_aligned:.3f}")
    print()

    one_hop_ceiling = (1 + DEGREE) / GROUP_SIZE
    print("MEMORY / COMPUTE TRADE")
    print(
        f"A same-memory one-hop table can expose at most {1+DEGREE}/{GROUP_SIZE} = "
        f"{one_hop_ceiling:.1%} of a group from one cue, below the {TARGET_RECALL:.0%} target."
    )
    print("Multi-hop local structure shares associations transitively, but a wider direct table")
    print("could trade more persistent memory for still less query computation.")
    print()

    print("INTERPRETATION")
    print("Repeated receiver-labelled co-use can be compiled into bounded local adjacency so")
    print("future work stays near the cue. But there is no single universal locality when two")
    print("receivers impose incompatible partitions: the pooled graph approaches global-search")
    print("scaling, while receiver-specific overlays remain constant-work in this toy.")
    print()
    print("GUARDRAIL")
    print("This is associative graph learning / clustering, not a new learning theorem. The")
    print("count table is a generous offline learning instrument and is compiled away. A real")
    print("DifferentMachine result still needs bounded online plasticity, task loss rather than")
    print("known group-recall structure, strong retrieval/router baselines, adaptation after")
    print("regime change, and actual systems accounting.")


if __name__ == "__main__":
    main()
