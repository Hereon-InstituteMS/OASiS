"""Tier-2: a time-varying Dirichlet value must be re-condensed every step.

Claim: skfem time_dependent#3 -- for non-homogeneous time-varying BCs, update
the rhs and re-condense each step; "if rhs is condensed once and reused, the
boundary DOFs stay at their initial values across the time loop -- solution at
boundary diverges from the prescribed time-varying BC."

Measured on skfem 12.0.1, heat equation on a 16x16 MeshTri.init_tensor with
ElementTriP1, backward Euler, dt = 2e-3, 20 steps, with a wall temperature
ramping g(t) = t/T from 0 to 1:

  * CONFIRMED, and the exact shape matters.  condense(A, b, x=g, D=D) writes
    the prescribed values into the returned solution vector, so the boundary
    DOFs come back holding whatever g was AT THE MOMENT condense WAS CALLED.
    Freezing that call at step 0 pins every boundary DOF at the initial value
    for the whole run.
  * IT IS COMPLETELY SILENT.  No exception, no warning, the interior still
    solves, and the final field is smooth and finite.  The only way to see it
    is to compare the boundary DOFs against the prescribed g(t).
  * the interior is wrong too, not just the boundary: the whole solution is
    off, because the frozen lifting feeds the wrong data into the interior
    equations at every step.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

NX = 16
DT = 2e-3
NSTEPS = 20


def wall_value(step):
    """Ramp the wall temperature from 0 to 1 over the run."""
    return step / NSTEPS


def integrate(refresh_bc):
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    A = (M + DT * K).tocsr()
    u = ib.zeros()
    frozen = None
    msgs = []
    for step in range(1, NSTEPS + 1):
        xbc = ib.zeros()
        xbc[D] = wall_value(step if refresh_bc else 0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if refresh_bc:
                u = solve(*condense(A, M @ u, x=xbc, D=D))
            else:
                # the reported bug: condense ONCE, reuse the pieces
                if frozen is None:
                    frozen = condense(A, M @ u, x=xbc, D=D)
                    Ac, bc, xc, I = frozen
                else:
                    Ac, _, xc, I = frozen
                    bc = (M @ u)[I] - (A[I][:, D] @ xc[D])
                nxt = xc.copy()
                nxt[I] = solve(Ac, bc)
                u = nxt
            msgs += [str(c.message) for c in caught]
    return u, ib, D, sorted(set(msgs))


def main() -> int:
    ok = True
    u_ok, ib, D, w_ok = integrate(True)
    u_bad, _, _, w_bad = integrate(False)
    target = wall_value(NSTEPS)
    print(f"dofs_N={ib.N} boundary_dofs={len(D)} nsteps={NSTEPS}")
    print(f"prescribed_final_wall_value={target}")
    print(f"recondensed_boundary_min={float(u_ok[D].min()):.6f}")
    print(f"recondensed_boundary_max={float(u_ok[D].max()):.6f}")
    print(f"frozen_boundary_min={float(u_bad[D].min()):.6f}")
    print(f"frozen_boundary_max={float(u_bad[D].max()):.6f}")
    err_ok = float(np.abs(u_ok[D] - target).max())
    err_bad = float(np.abs(u_bad[D] - target).max())
    print(f"recondensed_boundary_error={err_ok:.3e}")
    print(f"frozen_boundary_error={err_bad:.3e}")
    print(f"recondensed_boundary_tracks_bc={err_ok < 1e-12}")
    print(f"frozen_boundary_tracks_bc={err_bad < 1e-12}")
    print(f"frozen_boundary_stuck_at_initial_value="
          f"{float(np.abs(u_bad[D]).max()) < 1e-12}")
    if err_ok >= 1e-12:
        print("FAIL: the re-condensed run did not honour the BC",
              file=sys.stderr)
        ok = False
    if err_bad < 1e-12:
        print("FAIL: the frozen-condense run DID track the BC, so reusing "
              "the condensation is safe after all", file=sys.stderr)
        ok = False

    # --- silence -------------------------------------------------------
    print(f"frozen_warnings={w_bad!r}")
    print(f"frozen_failure_is_silent={not w_bad}")
    print(f"frozen_solution_finite={bool(np.isfinite(u_bad).all())}")
    if w_bad:
        print(f"FAIL: the frozen run warned {w_bad!r}", file=sys.stderr)
        ok = False
    if not np.isfinite(u_bad).all():
        print("FAIL: the frozen run produced non-finite values, so it is "
              "not the silent shape documented here", file=sys.stderr)
        ok = False

    # --- the interior is wrong too --------------------------------------
    interior = np.setdiff1d(np.arange(ib.N), D)
    rel = float(np.linalg.norm(u_bad[interior] - u_ok[interior])
                / np.linalg.norm(u_ok[interior]))
    print(f"interior_relative_difference={rel:.4e}")
    print(f"interior_is_wrong_too={rel > 1e-2}")
    if rel <= 1e-2:
        print("FAIL: only the boundary was affected", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
