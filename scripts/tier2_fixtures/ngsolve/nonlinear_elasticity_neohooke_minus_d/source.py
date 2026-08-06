"""Tier-2: the missing "- d" in the Neo-Hookean energy is a pure constant, so it
shifts a.Energy() and NOTHING else.

Claim: ngsolve nonlinear_elasticity#2 -- W = 0.5*mu*(Tr(C) - d) - mu*ln(J)
+ 0.5*lam*ln(J)^2.  The catalog asserts that dropping the "- d" makes the
stress-free reference carry a non-zero initial stress so that "the first Newton
iterate runs off looking for a different equilibrium", and recommends the
W(F = I) = 0 sanity check.

Wrong variant: build the same Variation form with 0.5*mu*Tr(C) instead of
0.5*mu*(Tr(C) - d) and compare energy, residual at u = 0, Newton iteration count
and the converged displacement vector against the correct form.

Observed on NGSolve 6.2.2604 (unit_square maxh 0.4, VectorH1 order 2):
  * the W(F = I) = 0 half of the claim HOLDS -- a.Energy(0) is exactly 0.0 for
    the correct energy and 0.5*mu*d*|Omega| for the truncated one;
  * the CONSEQUENCE half does NOT -- 0.5*mu*d is an additive constant whose
    Variation is identically zero, so a.Apply(0, r) gives Norm(r) == 0.0 for
    BOTH forms, Newton takes the same number of iterations, and the two
    converged coefficient vectors are bitwise identical (inf-norm difference
    0.0).  There is no initial stress and Newton does not run off.
This fixture pins that behaviour so a future change in either direction is seen.
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
    Integrate,
    Mesh,
    Norm,
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
DIM = 2

# The wrong variant subtracts this instead of DIM.
D_WRONG = 0.0


def energy_form(fes, d_sub: float):
    u = fes.TrialFunction()
    ident = Id(DIM)
    F = ident + Grad(u)
    C = F.trans * F
    J = Det(F)
    W = 0.5 * MU * (Trace(C) - d_sub) - MU * log(J) + 0.5 * LAM * log(J) ** 2
    a = BilinearForm(fes, symmetric=True)
    a += Variation(W * dx)
    return a


def probe(mesh, fes, d_sub: float):
    a = energy_form(fes, d_sub)
    gfu = GridFunction(fes)
    res = gfu.vec.CreateVector()
    energy0 = a.Energy(gfu.vec)          # W integrated at F = I
    a.Apply(gfu.vec, res)
    res0 = Norm(res)                     # residual at the reference state
    gfu.Set(CoefficientFunction((0.05 * x, 0.0)),
            definedon=mesh.Boundaries("right"))
    status, numit = solvers.Newton(a, gfu, maxit=30, printing=False, maxerr=1e-10)
    return energy0, res0, status, numit, gfu.vec.FV().NumPy().copy()


def main() -> int:
    ok = True

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = VectorH1(mesh, order=2, dirichlet="left|right")
    area = Integrate(CoefficientFunction(1.0), mesh)
    print(f"form_type={type(energy_form(fes, DIM)).__name__}")

    e_ok, r_ok, st_ok, nit_ok, v_ok = probe(mesh, fes, DIM)
    e_bad, r_bad, st_bad, nit_bad, v_bad = probe(mesh, fes, D_WRONG)

    # --- WRONG variant: W(F = I) is no longer zero -------------------------
    print(f"correct_energy_at_identity_is_zero={e_ok == 0.0}")
    bad_nonzero = abs(e_bad) > 1e-6
    print(f"wrong_energy_at_identity_nonzero={bad_nonzero}")
    expected_shift = 0.5 * MU * DIM * area
    print("wrong_energy_equals_half_mu_d_area="
          f"{abs(e_bad - expected_shift) < 1e-8 * expected_shift}")
    if not (e_ok == 0.0 and bad_nonzero):
        print(f"FAIL: the W(F=I) signature is absent (ok={e_ok}, bad={e_bad})",
              file=sys.stderr)
        ok = False

    # --- but the consequence claimed by the catalog does NOT follow --------
    print(f"correct_residual_at_identity_is_zero={r_ok == 0.0}")
    print(f"wrong_residual_at_identity_is_zero={r_bad == 0.0}")
    print(f"wrong_variant_has_no_initial_stress={r_bad == 0.0}")
    print(f"newton_status_equal={st_ok == st_bad == 0}")
    print(f"newton_numit_equal={nit_ok == nit_bad}")
    sol_diff = float(np.abs(v_ok - v_bad).max())
    print(f"converged_solutions_bitwise_identical={sol_diff == 0.0}")
    if r_bad != 0.0 or nit_ok != nit_bad or sol_diff != 0.0:
        print("FAIL: the missing -d term now changes the residual/solution -- "
              "Variation() of an additive constant is no longer zero",
              file=sys.stderr)
        ok = False

    if st_ok != 0 or nit_ok <= 0:
        print(f"FAIL: reference Newton solve unhealthy (status={st_ok}, "
              f"numit={nit_ok})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
