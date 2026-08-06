"""Tier-2: dampfactor < 1 rescues a Newton step that is too far from equilibrium.

Claim: ngsolve nonlinear_elasticity#6 -- when the load step is aggressive enough
that Newton overshoots the convergence basin, dampfactor = 0.5 or 0.25 in
ngsolve.solvers.Newton restores convergence; alternatively reduce the load
increment.

Wrong variant: prescribe the whole 0.8 top displacement (80 % nominal strain) in
ONE undamped step (dampfactor = 1.0) on unit_square maxh 0.3, VectorH1 order 2,
dirichlet="left|top".

Observed on NGSolve 6.2.2604:
  * dampfactor = 1.0 -> NgException: UmfpackInverse: Numeric factorization
    failed.  Raising maxit from 60 to 300 does NOT help -- the tangent goes
    singular, it is not a "needs more iterations" failure.
  * dampfactor = 0.5 and 0.25 -> status 0, and max|u_y| equals the prescribed
    0.8 exactly.
  * the documented alternative (two warm-started 0.4 increments, undamped) also
    converges.
Caveat recorded for the catalog: on this configuration damping did NOT cost more
iterations; the counts are printed for the record but not pinned.
"""

from __future__ import annotations

import sys

import numpy as np
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    Det,
    GridFunction,
    Grad,
    Id,
    Mesh,
    Projector,
    Trace,
    Variation,
    VectorH1,
    dx,
    log,
    solvers,
    unit_square,
)

E, NU = 200.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))

MAXH = 0.3
ORDER = 2
BIG_DISP = 0.8

# The wrong variant runs undamped.
DAMP_WRONG = 1.0


def build():
    mesh = Mesh(unit_square.GenerateMesh(maxh=MAXH))
    fes = VectorH1(mesh, order=ORDER, dirichlet="left|top")
    u = fes.TrialFunction()
    ident = Id(2)
    F = ident + Grad(u)
    C = F.trans * F
    J = Det(F)
    W = 0.5 * MU * (Trace(C) - 2) - MU * log(J) + 0.5 * LAM * log(J) ** 2
    a = BilinearForm(fes, symmetric=True)
    a += Variation(W * dx)
    return mesh, fes, a


def one_shot(disp: float, damp: float, maxit: int = 60):
    """Prescribe the full displacement in a single Newton solve."""
    mesh, fes, a = build()
    gfu = GridFunction(fes)
    gfu.Set(CoefficientFunction((0.0, disp)), definedon=mesh.Boundaries("top"))
    try:
        status, numit = solvers.Newton(a, gfu, maxit=maxit, dampfactor=damp,
                                       printing=False, maxerr=1e-10)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"
    uy = float(np.abs(gfu.components[1].vec.FV().NumPy()).max())
    return status, numit, "" if abs(uy - disp) < 1e-6 else f"uy={uy}"


def stepped(disp: float, nsteps: int, damp: float):
    """Warm-started load stepping: overwrite only the constrained dofs."""
    mesh, fes, a = build()
    gfu = GridFunction(fes)
    scratch = GridFunction(fes)
    proj_dir = Projector(fes.FreeDofs(), False)
    statuses = []
    for step in range(1, nsteps + 1):
        d = step / nsteps * disp
        scratch.Set(CoefficientFunction((0.0, d)),
                    definedon=mesh.Boundaries("top"))
        gfu.vec.data += proj_dir * (scratch.vec - gfu.vec)
        try:
            status, _n = solvers.Newton(a, gfu, maxit=60, dampfactor=damp,
                                        printing=False, maxerr=1e-10)
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        statuses.append(status)
    uy = float(np.abs(gfu.components[1].vec.FV().NumPy()).max())
    return all(s == 0 for s in statuses) and abs(uy - disp) < 1e-6, ""


def main() -> int:
    ok = True

    # --- WRONG variant: full 80 % strain in one undamped step ---------------
    st_u, nit_u, msg_u = one_shot(BIG_DISP, DAMP_WRONG)
    print(f"undamped_msg={msg_u!r}")
    undamped_broke = "Numeric factorization failed" in msg_u
    print(f"undamped_one_shot_breaks={undamped_broke}")
    if not undamped_broke:
        print(f"FAIL: undamped one-shot step converged (status={st_u}, "
              f"numit={nit_u}) -- the pitfall did not reproduce",
              file=sys.stderr)
        ok = False

    # more iterations is NOT the fix
    _st, _nit, msg_big_maxit = one_shot(BIG_DISP, DAMP_WRONG, maxit=300)
    still_broken = "Numeric factorization failed" in msg_big_maxit
    print(f"undamped_still_breaks_with_maxit_300={still_broken}")
    if not still_broken:
        print("FAIL: raising maxit alone fixed the undamped step",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant 1: damped Newton -------------------------------------
    results = {}
    for damp in (0.5, 0.25):
        st, nit, msg = one_shot(BIG_DISP, damp)
        results[damp] = (st, nit, msg)
        print(f"damp{damp}_status_zero={st == 0}")
        print(f"damp{damp}_reaches_prescribed_disp={msg == '' and st == 0}")
        if st != 0 or msg != "":
            print(f"FAIL: dampfactor={damp} did not rescue the step "
                  f"(status={st}, msg={msg!r})", file=sys.stderr)
            ok = False
    print(f"damped_numits={[results[d][1] for d in (0.5, 0.25)]}")

    # --- RIGHT variant 2: reduce the increment instead ----------------------
    good, msg_s = stepped(BIG_DISP, 2, 1.0)
    print(f"two_undamped_increments_converge={good}")
    if not good:
        print(f"FAIL: the documented alternative (smaller increment) failed "
              f"({msg_s!r})", file=sys.stderr)
        ok = False

    # --- for the record: damping did not cost extra iterations here ---------
    st_ref, nit_ref, _m = one_shot(0.5, 1.0)
    st_d, nit_d, _m2 = one_shot(0.5, 0.25)
    if st_ref == 0 and st_d == 0:
        print(f"info_undamped_numit_at_disp0p5={nit_ref} "
              f"damped_numit_at_disp0p5={nit_d}")
        print(f"damping_costs_more_iters_when_both_work={nit_d > nit_ref}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
