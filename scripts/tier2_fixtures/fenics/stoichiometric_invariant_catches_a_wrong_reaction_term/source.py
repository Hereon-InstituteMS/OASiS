"""Tier-2 for fenics reaction_diffusion#9: use the stoichiometric conservation
law as the correctness check. For 2A <-> B with no-flux boundaries the
combination A + 2B is conserved to round-off independently of mesh, degree and
step size, because the constant test function (1, 2) lies in the mixed space and
the reaction terms cancel identically in the discrete residual.

Wrong variant here: the A equation carries r instead of 2r -- one wrong
stoichiometric factor, nothing else. It still solves, the SNES still converges,
and the species profiles still look plausible.

Observed on dolfinx 0.10.0 (16x16 unit square, P1 x P1, 25 backward-Euler
steps): with the correct factor the invariant assembled with
dolfinx.fem.assemble_scalar(dolfinx.fem.form((A_h + 2*B_h)*ufl.dx)) drifts by
about 1e-15 relative while the species ranges move a long way; with the wrong
factor it drifts by O(10 percent) -- immediately, and with no other complaint
from the solver.

Mutation control: T2_MUTATE=1 restores the correct stoichiometric factor.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, D, KF, KR, DT, NSTEP = 16, 0.01, 1.0, 1.0, 0.05, 25


def run(correct: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    w = dolfinx.fem.Function(W)
    w_n = dolfinx.fem.Function(W)
    w_n.sub(0).interpolate(lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x[0]))
    w_n.sub(1).interpolate(lambda x: np.full_like(x[0], 0.2))
    w_n.x.scatter_forward()
    w.x.array[:] = w_n.x.array
    A, B = ufl.split(w)
    An, Bn = ufl.split(w_n)
    va, vb = ufl.TestFunctions(W)
    r = KF * A * A - KR * B
    nu_a = 2.0 if correct else 1.0  # stoichiometric factor of A in 2A <-> B
    F = (((A - An) / DT) * va * ufl.dx
         + D * ufl.dot(ufl.grad(A), ufl.grad(va)) * ufl.dx
         + nu_a * r * va * ufl.dx
         + ((B - Bn) / DT) * vb * ufl.dx
         + D * ufl.dot(ufl.grad(B), ufl.grad(vb)) * ufl.dx
         - r * vb * ufl.dx)
    inv = dolfinx.fem.form((A + 2 * B) * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, w, petsc_options_prefix=f"t2_rd9_{'ok' if correct else 'bad'}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    a_map = np.array(W.sub(0).collapse()[1], dtype=np.int32)
    b_map = np.array(W.sub(1).collapse()[1], dtype=np.int32)
    start = (float(w.x.array[a_map].min()), float(w.x.array[a_map].max()),
             float(w.x.array[b_map].min()), float(w.x.array[b_map].max()))
    inv0 = float(dolfinx.fem.assemble_scalar(inv))
    reasons = []
    for _ in range(NSTEP):
        prob.solve()
        w.x.scatter_forward()
        reasons.append(int(prob.solver.getConvergedReason()))
        w_n.x.array[:] = w.x.array
    inv1 = float(dolfinx.fem.assemble_scalar(inv))
    end = (float(w.x.array[a_map].min()), float(w.x.array[a_map].max()),
           float(w.x.array[b_map].min()), float(w.x.array[b_map].max()))
    return inv0, inv1, start, end, reasons


def main() -> int:
    i0_t, i1_t, s_t, e_t, rea_t = run(correct=MUTATE)
    i0_r, i1_r, s_r, e_r, rea_r = run(correct=True)
    drift_t = abs(i1_t - i0_t) / abs(i0_t)
    drift_r = abs(i1_r - i0_r) / abs(i0_r)

    print(f"correct_stoichiometry: invariant {i0_r:.12f} -> {i1_r:.12f} "
          f"relative_drift={drift_r:.3e}")
    print(f"correct_stoichiometry: A [{s_r[0]:.3f}, {s_r[1]:.3f}] -> "
          f"[{e_r[0]:.3f}, {e_r[1]:.3f}]  B [{s_r[2]:.3f}, {s_r[3]:.3f}] -> "
          f"[{e_r[2]:.3f}, {e_r[3]:.3f}]")
    print(f"under_test: invariant {i0_t:.12f} -> {i1_t:.12f} "
          f"relative_drift={drift_t:.3e} snes_reasons_all_positive="
          f"{all(r > 0 for r in rea_t)}")

    moved = max(abs(a - b) for a, b in zip(e_r, s_r)) > 0.1
    print(f"species_ranges_really_moved={moved}")
    print(f"correct_stoichiometry_holds_the_invariant_to_roundoff="
          f"{drift_r < 1e-12}")
    print(f"wrong_factor_still_converged={all(r > 0 for r in rea_t)}")
    print(f"wrong_factor_drifts_by_more_than_a_percent={drift_t > 0.01}")

    if (drift_r < 1e-12 and moved and drift_t > 0.01
            and all(r > 0 for r in rea_t)):
        print("VERDICT=the_invariant_catches_the_wrong_stoichiometric_factor")
        return 0
    print("VERDICT=the_invariant_did_not_separate_them")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
