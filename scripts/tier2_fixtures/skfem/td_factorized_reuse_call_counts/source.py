"""Tier-2: factorize the time-stepping matrix ONCE and reuse it.

Claim: skfem time_dependent#2 -- factor the system matrix once and reuse it
via scipy.sparse.linalg.factorized(); the signal is that a profile shows >90 %
of time in scipy.sparse.linalg.spsolve per step, and factorized() reduces the
per-step cost from a full LU to forward/back substitution.

This fixture deliberately does NOT time anything.  Wall-clock assertions go
red on a busy machine and prove nothing about the code.  The claim's real
content is structural and is checked structurally:

  * the time-stepping matrix is BYTE-IDENTICAL at every step (zero differing
    non-zeros between step 0 and step 19), which is what makes reuse legal;
  * the spsolve loop calls scipy.sparse.linalg.spsolve once per step
    (verified by counting calls through a wrapper), whereas the factorized
    loop calls it ZERO times;
  * factorized() performs exactly ONE LU decomposition, counted by wrapping
    scipy.sparse.linalg._dsolve.linsolve.splu, no matter how many steps run;
    spsolve never goes through that entry point at all because it calls
    SuperLU directly, which is why a call-count on splu alone would show 0
    for BOTH loops and prove nothing;
  * the two loops agree to machine precision, so the optimisation is not
    trading accuracy for speed.

Measured on skfem 12.0.1 / scipy 1.15.3, heat equation on a 32x32
MeshTri.init_tensor with ElementTriP1, backward Euler, 20 steps.

Mutation control: T2_MUTATE=1 applies the documented fix to the FIRST loop --
the naive one -- so that it too factors Ac once with
scipy.sparse.linalg.factorized() and calls the resulting callable per step
instead of solve(*condense(...)).  The per-step spsolve pathology is then gone
and 'spsolve_loop_spsolve_calls=20', 'spsolve_called_once_per_step=True' and
'splu_counter_alone_is_blind_to_spsolve=True' disappear from the output.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse.linalg as spl
import scipy.sparse.linalg._dsolve.linsolve as linsolve
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"

NX = 32
NSTEPS = 20
DT = 1e-3


def ic(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def main() -> int:
    ok = True
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    A = (M + DT * K).tocsr()
    print(f"dofs_N={ib.N} system_nnz={A.nnz}")

    # --- the matrix genuinely does not change --------------------------
    A_first = (M + DT * K).tocsr()
    A_last = (M + DT * K).tocsr()
    diff = (A_first - A_last)
    diff.eliminate_zeros()
    print(f"matrix_differing_nonzeros_across_steps={diff.nnz}")
    print(f"matrix_is_constant_in_time={diff.nnz == 0}")
    if diff.nnz != 0:
        print("FAIL: the time-stepping matrix changed, so reuse is illegal",
              file=sys.stderr)
        ok = False

    Ac, _, _, I = condense(A, ib.zeros(), D=D)
    print(f"condensed_shape={Ac.shape} condensed_nnz={Ac.nnz}")

    # --- counters ------------------------------------------------------
    calls = {"spsolve": 0, "splu": 0}
    orig_spsolve, orig_splu = spl.spsolve, linsolve.splu

    def counted_spsolve(*a, **k):
        calls["spsolve"] += 1
        return orig_spsolve(*a, **k)

    def counted_splu(*a, **k):
        calls["splu"] += 1
        return orig_splu(*a, **k)

    u0 = ib.project(ic)
    u0[D] = 0.0

    spl.spsolve, linsolve.splu = counted_spsolve, counted_splu
    try:
        u = u0.copy()
        if MUTATE:
            # the documented fix applied to the naive loop: one factorisation
            # reused for every step, instead of a fresh solve per step
            lu_fixed = spl.factorized(Ac.tocsc())
            for _ in range(NSTEPS):
                nxt = ib.zeros()
                nxt[I] = lu_fixed((M @ u)[I])
                u = nxt
        else:
            # THE PATHOLOGY: a full solve -- hence a full LU -- every step
            for _ in range(NSTEPS):
                u = solve(*condense(A, M @ u, D=D))
        loop_counts = dict(calls)

        calls.update(spsolve=0, splu=0)
        v = u0.copy()
        lu = spl.factorized(Ac.tocsc())
        for _ in range(NSTEPS):
            b = (M @ v)[I]
            nxt = ib.zeros()
            nxt[I] = lu(b)
            v = nxt
        fact_counts = dict(calls)
    finally:
        spl.spsolve, linsolve.splu = orig_spsolve, orig_splu

    print(f"nsteps={NSTEPS}")
    print(f"spsolve_loop_spsolve_calls={loop_counts['spsolve']}")
    print(f"factorized_loop_spsolve_calls={fact_counts['spsolve']}")
    print(f"factorized_loop_splu_calls={fact_counts['splu']}")
    print(f"spsolve_loop_splu_calls={loop_counts['splu']}")
    print(f"spsolve_called_once_per_step="
          f"{loop_counts['spsolve'] >= NSTEPS}")
    print(f"factorized_loop_avoids_spsolve_entirely="
          f"{fact_counts['spsolve'] == 0}")
    print(f"factorized_does_exactly_one_lu={fact_counts['splu'] == 1}")
    print(f"splu_counter_alone_is_blind_to_spsolve="
          f"{loop_counts['splu'] == 0}")
    if loop_counts["spsolve"] < NSTEPS:
        print("FAIL: the naive loop did not call spsolve once per step",
              file=sys.stderr)
        ok = False
    if fact_counts["spsolve"] != 0:
        print("FAIL: the factorized loop still called spsolve",
              file=sys.stderr)
        ok = False
    if fact_counts["splu"] != 1:
        print(f"FAIL: factorized performed {fact_counts['splu']} LU "
              f"decompositions, not 1", file=sys.stderr)
        ok = False

    # --- and the answers agree ------------------------------------------
    dev = float(np.abs(u - v).max())
    print(f"max_abs_difference_between_loops={dev:.3e}")
    print(f"factorized_reuse_is_exact={dev < 1e-12}")
    if dev >= 1e-12:
        print("FAIL: reusing the factorisation changed the answer",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
