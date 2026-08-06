"""Tier-2: the explicit wave update must re-zero the Dirichlet DOFs each step.

Claim: skfem wave#5 -- Dirichlet BCs must be re-applied every step
(u_new[bnd_dofs] = 0); the explicit update propagates non-zero values into
boundary DOFs via the off-diagonal K coupling, and "without the
re-application, energy slowly leaks at the boundary and the simulation is no
longer a homogeneous-BC problem.  Signal: |u| at boundary DOFs grows from 0 to
non-negligible values (>1e-6) within a few hundred steps."

Measured on skfem 12.0.1, MeshQuad.init_tensor 21x21, ElementQuad1, c = 1,
row-sum-lumped mass, standing-wave IC, dt = 0.5 * h/c, 800 steps:

  * CONFIRMED, and the threshold in the entry is far too generous.  The
    boundary does not creep to 1e-6 -- it reaches O(1) within the first few
    dozen steps, because the very first update writes -dt^2 c^2 (K u)/M_L
    into every boundary DOF and K has non-zero rows there.
  * IT IS SILENT: no warning, no exception, the solution stays finite and
    keeps oscillating, so nothing in the run says the boundary condition has
    stopped being enforced.
  * the interior is wrong too: with the boundary left free the interior
    solution diverges from the properly constrained one by an O(1) relative
    amount over the run.
  * with the re-application the boundary DOFs stay EXACTLY zero -- not
    small, zero -- which makes the check trivial to write.

Mutation control: T2_MUTATE=1 applies the documented fix to the pathological
second run -- run(reapply=True), i.e. un[D] = 0.0 after every update instead
of leaving the boundary DOFs free.  The boundary then never leaves zero, so
'free_boundary_exceeds_1e_minus_6=True', 'free_boundary_reaches_order_one=True',
'free_boundary_order_one_within_100_steps=True' and 'interior_is_wrong_too=True'
disappear from the output.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from skfem import Basis, ElementQuad1, MeshQuad
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"

C = 1.0
NX = 21
NSTEPS = 800


def run(reapply):
    m = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX),
                             np.linspace(0.0, 1.0, NX))
    ib = Basis(m, ElementQuad1())
    K = laplace.assemble(ib).tocsr()
    M = mass.assemble(ib).tocsr()
    D = ib.get_dofs().all()
    ML = np.asarray(M.sum(axis=1)).ravel()
    h = 1.0 / (NX - 1)
    dt = 0.5 * h / C

    def acc(u):
        return (-C ** 2 * (K @ u)) / ML

    u0 = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u0[D] = 0.0
    um = u0 + 0.5 * dt * dt * acc(u0)
    if reapply:
        um[D] = 0.0
    u = u0.copy()
    bmax = [float(np.abs(u[D]).max())]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(NSTEPS):
            un = 2.0 * u - um + dt * dt * acc(u)
            if reapply:
                un[D] = 0.0
            um, u = u, un
            bmax.append(float(np.abs(u[D]).max()))
        msgs = sorted({str(c.message) for c in caught})
    return u, ib, D, np.array(bmax), msgs


def main() -> int:
    ok = True
    u_ok, ib, D, b_ok, m_ok = run(True)
    # THE PATHOLOGY: reapply=False leaves the Dirichlet DOFs free after each
    # explicit update.  Under T2_MUTATE=1 the documented fix is applied here
    # and this run re-zeroes un[D] every step as well.
    u_bad, _, _, b_bad, m_bad = run(True if MUTATE else False)
    print(f"dofs_N={ib.N} boundary_dofs={len(D)} steps={NSTEPS}")

    print(f"reapplied_boundary_max_over_run={b_ok.max():.6e}")
    print(f"reapplied_boundary_is_exactly_zero={bool(b_ok.max() == 0.0)}")
    print(f"free_boundary_max_over_run={b_bad.max():.6e}")
    print(f"free_boundary_final={b_bad[-1]:.6e}")
    print(f"free_boundary_exceeds_1e_minus_6={bool(b_bad.max() > 1e-6)}")
    print(f"free_boundary_reaches_order_one={bool(b_bad.max() > 0.1)}")
    if b_ok.max() != 0.0:
        print("FAIL: the re-applied run did not hold the boundary at zero",
              file=sys.stderr)
        ok = False
    if not b_bad.max() > 1e-6:
        print("FAIL: the boundary stayed below 1e-6 without re-application",
              file=sys.stderr)
        ok = False

    # --- how quickly does it leave zero? --------------------------------
    first = int(np.argmax(b_bad > 1e-6))
    reaches_1 = int(np.argmax(b_bad > 0.1)) if (b_bad > 0.1).any() else -1
    print(f"free_boundary_first_step_above_1e_minus_6={first}")
    print(f"free_boundary_first_step_above_0p1={reaches_1}")
    print(f"free_boundary_leaves_zero_immediately={first <= 2}")
    print(f"free_boundary_order_one_within_100_steps="
          f"{0 <= reaches_1 <= 100}")
    if first > 2:
        print(f"FAIL: the boundary took {first} steps to leave zero",
              file=sys.stderr)
        ok = False

    # --- silence and the interior ----------------------------------------
    print(f"free_boundary_warnings={m_bad!r}")
    print(f"free_boundary_run_is_silent={not m_bad}")
    print(f"free_boundary_solution_finite={bool(np.isfinite(u_bad).all())}")
    interior = np.setdiff1d(np.arange(ib.N), D)
    rel = float(np.linalg.norm(u_bad[interior] - u_ok[interior])
                / np.linalg.norm(u_ok[interior]))
    print(f"interior_relative_difference={rel:.4e}")
    print(f"interior_is_wrong_too={rel > 0.1}")
    if m_bad:
        print(f"FAIL: the un-constrained run warned {m_bad!r}",
              file=sys.stderr)
        ok = False
    if not np.isfinite(u_bad).all():
        print("FAIL: the un-constrained run went non-finite, so it is not "
              "the silent shape documented here", file=sys.stderr)
        ok = False
    if rel <= 0.1:
        print("FAIL: only the boundary was affected", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
