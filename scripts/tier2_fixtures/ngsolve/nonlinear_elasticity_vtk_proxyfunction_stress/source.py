"""Tier-2: VTKOutput cannot evaluate a stress built from fes.TrialFunction().

Claim: ngsolve nonlinear_elasticity#7 -- after Newton converges, the Cauchy/PK1
stress handed to VTKOutput must be rebuilt from the resolved GridFunction
(Grad(gfu), Det(I+Grad(gfu)), Inv(...)), NOT from the symbolic
fes.TrialFunction().  The symbolic version is a ProxyFunction and VTKOutput.Do()
raises NgException 'cannot evaluate ProxyFunction without userdata'.

Wrong variant: build sigma from u = fes.TrialFunction() and pass it through
coefs=[gfu, sigma_cauchy], exactly as the catalog nonlinear_elasticity_2d
template shapes its VTK block.

Observed on NGSolve 6.2.2604: VTKOutput(...) constructs fine -- the failure is
deferred to .Do(), which raises
    NgException: cannot evaluate ProxyFunction without userdata
Rebuilding the identical algebra from gfu writes the file without complaint.
"""

from __future__ import annotations

import sys

from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    Det,
    GridFunction,
    Grad,
    Id,
    Inv,
    Mesh,
    Trace,
    Variation,
    VectorH1,
    VTKOutput,
    dx,
    log,
    solvers,
    unit_square,
    x,
)

E, NU = 200.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))


def pk2(C, J):
    """Neo-Hookean 2nd Piola-Kirchhoff stress S = dW/dE."""
    return MU * Id(2) - MU / J**2 * Inv(C) + LAM * log(J) / J**2 * Inv(C)


def main() -> int:
    ok = True

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = VectorH1(mesh, order=2, dirichlet="left|right")
    u = fes.TrialFunction()

    ident = Id(2)
    F = ident + Grad(u)
    C = F.trans * F
    J = Det(F)
    energy = 0.5 * MU * (Trace(C) - 2) - MU * log(J) + 0.5 * LAM * log(J) ** 2
    a = BilinearForm(fes, symmetric=True)
    a += Variation(energy * dx)

    gfu = GridFunction(fes)
    gfu.Set(CoefficientFunction((0.05 * x, 0.0)), definedon=mesh.Boundaries("right"))
    status, numit = solvers.Newton(a, gfu, maxit=25, printing=False, maxerr=1e-10)
    print(f"newton_status={status} newton_numit_positive={numit > 0}")
    if status != 0 or numit <= 0:
        print(f"FAIL: reference Newton solve did not converge (status={status})",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: stress built from the ProxyFunction -----------------
    print(f"trialfunction_type={type(u).__name__}")
    F_sym = ident + Grad(u)
    C_sym = F_sym.trans * F_sym
    J_sym = Det(F_sym)
    sigma_sym = 1 / J_sym * F_sym * pk2(C_sym, J_sym) * F_sym.trans

    ctor_ok = True
    msg = ""
    try:
        vtk_bad = VTKOutput(mesh, coefs=[gfu, sigma_sym],
                            names=["displacement", "cauchy_stress"],
                            filename="proxy_bad", subdivision=0)
    except Exception as exc:  # noqa: BLE001
        ctor_ok = False
        msg = f"{type(exc).__name__}: {exc}"
    else:
        try:
            vtk_bad.Do()
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
    print(f"vtk_ctor_accepted_proxy_stress={ctor_ok}")
    print(f"proxy_do_msg={msg!r}")
    raised = "cannot evaluate ProxyFunction without userdata" in msg
    print(f"proxy_stress_raises_userdata_error={raised}")
    if not raised:
        print(f"FAIL: symbolic ProxyFunction stress did not raise the userdata "
              f"error (msg={msg!r})", file=sys.stderr)
        ok = False

    # --- RIGHT variant: rebuild the same algebra from gfu -------------------
    F_ev = ident + Grad(gfu)
    C_ev = F_ev.trans * F_ev
    J_ev = Det(F_ev)
    sigma_ev = 1 / J_ev * F_ev * pk2(C_ev, J_ev) * F_ev.trans
    good_msg = ""
    try:
        VTKOutput(mesh, coefs=[gfu, sigma_ev],
                  names=["displacement", "cauchy_stress"],
                  filename="proxy_good", subdivision=0).Do()
    except Exception as exc:  # noqa: BLE001
        good_msg = f"{type(exc).__name__}: {exc}"
    print(f"gridfunction_stress_writes_ok={not good_msg}")
    if good_msg:
        print(f"FAIL: the documented fix (rebuild from gfu) also failed: "
              f"{good_msg!r}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
