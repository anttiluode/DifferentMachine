from __future__ import annotations
import time
import numpy as np
from different_machine.core import MachineSpec, EventMachine, ClockedMachine, generate_events


def main() -> None:
    T = 20_000
    spec = MachineSpec.random(n_contacts=4096, n_branches=64, n_receivers=2, seed=7)
    events = generate_events(T, spec.n_contacts, event_probability=0.02, seed=11)

    t0 = time.perf_counter()
    dense = ClockedMachine(spec)
    y_dense, cand_dense = dense.run(events, T)
    dense_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    lazy = EventMachine(spec)
    y_lazy, cand_lazy = lazy.run(events, T)
    lazy_s = time.perf_counter() - t0

    cand_err = max(
        (abs(a.value - b.value) for a, b in zip(cand_dense, cand_lazy)),
        default=0.0,
    )
    y_err = float(np.max(np.abs(y_dense - y_lazy)))

    dense_private = dense.contact_touches + dense.branch_touches
    lazy_private = lazy.contact_touches + lazy.branch_touches

    print("DifferentMachine Gate 0 — SAME MACHINE, DIFFERENT EXECUTION")
    print(f"contacts                       : {spec.n_contacts}")
    print(f"branches                       : {spec.n_branches}")
    print(f"receivers                      : {spec.n_receivers}")
    print(f"ticks                          : {T}")
    print(f"input events                   : {len(events)} ({len(events)/T:.3%} of ticks)")
    print(f"max candidate mismatch         : {cand_err:.3e}")
    print(f"max final receiver mismatch    : {y_err:.3e}")
    print()
    print("PRIVATE-STATE TOUCHES")
    print(f"clocked                        : {dense_private:,}")
    print(f"addressed/lazy                 : {lazy_private:,}")
    print(f"logical touch ratio            : {dense_private/max(lazy_private,1):,.1f}x")
    print()
    print("REFERENCE PYTHON/NUMPY TIME")
    print(f"clocked                        : {dense_s:.4f} s")
    print(f"addressed/lazy                 : {lazy_s:.4f} s")
    print(f"ratio                          : {dense_s/max(lazy_s,1e-12):.2f}x")
    print()
    print("PASS" if cand_err < 1e-12 and y_err < 1e-12 else "FAIL")
    print("Quiet state persists without a full recurrent sweep. The wall-clock ratio is")
    print("a reference implementation result, not a hardware or GPU claim.")


if __name__ == "__main__":
    main()
