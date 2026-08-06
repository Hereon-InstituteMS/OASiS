"""Tier-2: an inverted initial guess makes det(F) negative, ln(J) NaN and the
first Newton linearisation unfactorisable.

Claim: ngsolve nonlinear_elasticity#0 -- det(F) must stay > 0; an initial
displacement guess that crushes the element to near-zero or negative volume
(e.g. u = -2*x on a unit cell) evaluates ln(J) with J <= 0 and produces
FloatingPointError or NaN in the first Newton residual.  Start from u = 0 and
load-step.

Wrong variant: gfu.Set(CF((-2*x, 0))) before the first Newton call, i.e. F =
diag(-1, 1), det F = -1.

Observed on NGSolve 6.2.2604: no FloatingPointError -- NGSolve returns NaN
silently.  Integrate(Det(F)) = -1, a.Energy(gfu.vec) is nan, a.Apply fills the
residual with NaN (Norm(r) is nan), and solvers.Newton then aborts with
    NgException: UmfpackInverse: Numeric factorization failed.
Starting from u = 0 and load-stepping the same compression to -0.4 keeps
Det(F) > 0 over the whole domain and converges at every step.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    Det,
    GridFunction,
    Grad,
    Id,
    IfPos,
    Integrate,
    Mesh,
    Norm,
    Projector,
    Trace,
    Variation,
    VectorH1,
    dx,
    log,
    solvers,
    unit_square,
    x,
)

E, NU = 200.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))

# u_x = CRUSH_FACTOR * x  as the initial guess.  -2 inverts the cell.
CRUSH_FACTOR = -2.0


def make():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = VectorH1(mesh, order=1, dirichlet="left|right")
    u = fes.TrialFunction()
    ident = Id(2)
    F = ident + Grad(u)
    C = F.trans * F
    J = Det(F)
    W = 0.5 * MU * (Trace(C) - 2) - MU * log(J) + 0.5 * LAM * log(J) ** 2
    a = BilinearForm(fes, symmetric=True)
    a += Variation(W * dx)
    return mesh, fes, a


def bad_area_fraction(mesh, gfu) -> float:
    """Area fraction of the mesh where Det(I + Grad(gfu)) <= 0."""
    J = Det(Id(2) + Grad(gfu))
    return float(Integrate(IfPos(J - 1e-12, 0.0, 1.0), mesh))


def main() -> int:
    ok = True

    # --- WRONG variant: inverted initial guess ------------------------------
    mesh, fes, a = make()
    gfu = GridFunction(fes)
    gfu.Set(CoefficientFunction((CRUSH_FACTOR * x, 0.0)))
    mean_J = Integrate(Det(Id(2) + Grad(gfu)), mesh)
    print(f"crushed_mean_detF_le_zero={mean_J <= 0.0}")
    print(f"crushed_inverted_area_fraction_is_one="
          f"{abs(bad_area_fraction(mesh, gfu) - 1.0) < 1e-9}")

    energy = a.Energy(gfu.vec)
    res = gfu.vec.CreateVector()
    a.Apply(gfu.vec, res)
    res_nan = bool(np.isnan(res.FV().NumPy()).any())
    print(f"crushed_energy_is_nan={math.isnan(energy)}")
    print(f"crushed_residual_has_nan={res_nan}")
    print(f"crushed_residual_norm_is_nan={math.isnan(Norm(res))}")
    if not (math.isnan(energy) and res_nan):
        print(f"FAIL: inverted guess did not poison the energy/residual "
              f"(energy={energy}, res_nan={res_nan})", file=sys.stderr)
        ok = False

    fpe = False
    msg = ""
    try:
        solvers.Newton(a, gfu, maxit=5, printing=False, maxerr=1e-10)
    except FloatingPointError as exc:
        fpe = True
        msg = f"FloatingPointError: {exc}"
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
    print(f"crushed_raises_floatingpointerror={fpe}")
    print(f"crushed_newton_msg={msg!r}")
    newton_broke = "Numeric factorization failed" in msg
    print(f"crushed_newton_factorization_failed={newton_broke}")
    if not newton_broke:
        print(f"FAIL: Newton from the inverted guess did not abort "
              f"(msg={msg!r})", file=sys.stderr)
        ok = False

    # --- RIGHT variant: from u = 0, load-step the SAME compression ----------
    mesh2, fes2, a2 = make()
    gfu2 = GridFunction(fes2)
    scratch = GridFunction(fes2)
    proj_dir = Projector(fes2.FreeDofs(), False)
    total, nsteps = -0.4, 4
    statuses = []
    for step in range(1, nsteps + 1):
        d = step / nsteps * total
        scratch.Set(CoefficientFunction((d, 0.0)),
                    definedon=mesh2.Boundaries("right"))
        gfu2.vec.data += proj_dir * (scratch.vec - gfu2.vec)
        status, _numit = solvers.Newton(a2, gfu2, maxit=25, printing=False,
                                        maxerr=1e-10)
        statuses.append(status)
    frac_bad = bad_area_fraction(mesh2, gfu2)
    print(f"loadstep_all_converged={all(s == 0 for s in statuses)}")
    print(f"loadstep_detF_positive_everywhere={frac_bad == 0.0}")
    print(f"loadstep_energy_finite={math.isfinite(a2.Energy(gfu2.vec))}")
    if not (all(s == 0 for s in statuses) and frac_bad == 0.0):
        print(f"FAIL: load stepping from u=0 did not stay admissible "
              f"(statuses={statuses}, bad_frac={frac_bad})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
