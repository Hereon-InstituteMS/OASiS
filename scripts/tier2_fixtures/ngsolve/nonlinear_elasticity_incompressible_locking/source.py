"""Tier-2: volumetric locking of a pure-displacement VectorH1 solve at
nu = 0.4999, measured on Cook's membrane against a mixed (u, p) reference.

Claim: ngsolve nonlinear_elasticity#4 -- for nearly incompressible materials use
F-bar or a mixed (u, p) formulation.  The catalog's stated signal is that a
pure-displacement VectorH1 solve at nu = 0.4999 gives a Cook-membrane tip
displacement "< 1% of the analytic value", and that mixed Taylor-Hood-like spaces
(VectorH1 order 2 + H1 order 1 for p) recover "within ~1%".

Wrong variant: pure-displacement Neo-Hookean on Cook's membrane
(SplineGeometry (0,0)-(48,44)-(48,60)-(0,44), maxh 4, left clamped, shear
traction on the right edge) with VectorH1 order 1 at nu = 0.4999.
Reference: the perturbed-Lagrangian mixed energy
    W = 0.5*mu*(Tr(C) - 2) - mu*ln(J) + p*(J - 1) - p^2/(2*lam)
on FESpace([VectorH1(order=2), H1(order=1)]).

Observed on NGSolve 6.2.2604:
  * nu = 1/3   -- pure P1 1.806, pure P2 1.8655, mixed 1.8654 (P2 and mixed
    agree to ~2e-5 relative);
  * nu = 0.4999 -- pure P1 0.8838, i.e. ~55% of the mixed 1.6141: the locking is
    real and large;  pure P2 1.6091 is within 0.4% of mixed.
So the mixed half of the claim holds, but "< 1% of the analytic value" is FALSE
by nearly two orders of magnitude, and the order-2 pure-displacement space the
catalog templates actually use barely locks at all on this mesh.  Both facts are
pinned below.
"""

from __future__ import annotations

import sys

from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    Det,
    FESpace,
    GridFunction,
    Grad,
    H1,
    Id,
    Mesh,
    Trace,
    Variation,
    VectorH1,
    ds,
    dx,
    log,
    solvers,
)

E = 70.0
SHEAR = 6.25 / 16.0          # total 6.25 spread over the 16-long right edge
MAXH = 4.0
NU_INCOMP = 0.4999
NU_MILD = 1.0 / 3.0

# Order of the pure-displacement space used for the locking variant.
LOCK_ORDER = 1


def lame(nu: float) -> tuple[float, float]:
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def cook_mesh() -> Mesh:
    geo = SplineGeometry()
    pts = [(0, 0), (48, 44), (48, 60), (0, 44)]
    pi = [geo.AppendPoint(*p) for p in pts]
    for a, b, name in ((0, 1, "bottom"), (1, 2, "right"),
                       (2, 3, "top"), (3, 0, "left")):
        geo.Append(["line", pi[a], pi[b]], bc=name)
    return Mesh(geo.GenerateMesh(maxh=MAXH))


def solve_pure(mesh, nu: float, order: int):
    mu, lam = lame(nu)
    fes = VectorH1(mesh, order=order, dirichlet="left")
    u = fes.TrialFunction()
    F = Id(2) + Grad(u)
    C = F.trans * F
    J = Det(F)
    W = 0.5 * mu * (Trace(C) - 2) - mu * log(J) + 0.5 * lam * log(J) ** 2
    a = BilinearForm(fes, symmetric=True)
    a += Variation(W * dx)
    a += Variation((-SHEAR * u[1]) * ds(definedon=mesh.Boundaries("right")))
    gfu = GridFunction(fes)
    status, _n = solvers.Newton(a, gfu, maxit=40, printing=False, maxerr=1e-9)
    return status, gfu


def solve_mixed(mesh, nu: float):
    mu, lam = lame(nu)
    V = VectorH1(mesh, order=2, dirichlet="left")
    Q = H1(mesh, order=1)
    fes = FESpace([V, Q])
    (u, p), (_v, _q) = fes.TnT()
    F = Id(2) + Grad(u)
    C = F.trans * F
    J = Det(F)
    W = 0.5 * mu * (Trace(C) - 2) - mu * log(J) + p * (J - 1) - p * p / (2 * lam)
    a = BilinearForm(fes, symmetric=True)
    a += Variation(W * dx)
    a += Variation((-SHEAR * u[1]) * ds(definedon=mesh.Boundaries("right")))
    gf = GridFunction(fes)
    status, _n = solvers.Newton(a, gf, maxit=40, printing=False, maxerr=1e-9)
    return status, gf, fes


def tip(mesh, disp) -> float:
    return float(disp(mesh(47.999, 52.0))[1])


def main() -> int:
    ok = True

    mesh = cook_mesh()
    print(f"cook_boundaries_named={sorted(set(mesh.GetBoundaries()))}")

    st_m, gf_m, fes_m = solve_mixed(mesh, NU_INCOMP)
    print(f"mixed_space_type={type(fes_m).__name__} "
          f"mixed_components={[type(c).__name__ for c in fes_m.components]}")
    ref = tip(mesh, gf_m.components[0])
    print(f"mixed_status_zero={st_m == 0} mixed_tip_positive={ref > 0}")
    if st_m != 0 or ref <= 0:
        print(f"FAIL: mixed reference solve unhealthy (status={st_m}, "
              f"tip={ref})", file=sys.stderr)
        ok = False

    # --- WRONG variant: pure displacement, low order, near-incompressible ---
    st_l, gf_l = solve_pure(mesh, NU_INCOMP, LOCK_ORDER)
    locked = tip(mesh, gf_l)
    ratio_locked = locked / ref
    print(f"locked_status_zero={st_l == 0}")
    print(f"locked_ratio_lt_0p6={ratio_locked < 0.6}")
    print(f"locked_loses_more_than_a_third={ratio_locked < 2.0 / 3.0}")
    if not (st_l == 0 and ratio_locked < 0.6):
        print(f"FAIL: pure-displacement order {LOCK_ORDER} did not lock at "
              f"nu={NU_INCOMP} (ratio={ratio_locked:.4f})", file=sys.stderr)
        ok = False

    # the catalog's "< 1% of the value" is not what happens
    print(f"locked_ratio_below_1pct={ratio_locked < 0.01}")
    print(f"locked_ratio_above_10pct={ratio_locked > 0.10}")
    if ratio_locked < 0.01:
        print("FAIL: locking really did collapse below 1% -- revisit the "
              "falsification recorded in this fixture", file=sys.stderr)
        ok = False

    # --- same space, mild nu: no locking -> the effect is nu-driven ---------
    st_mild, gf_mild = solve_pure(mesh, NU_MILD, LOCK_ORDER)
    _st_ref_mild, gf_ref_mild, _f = solve_mixed(mesh, NU_MILD)
    ratio_mild = tip(mesh, gf_mild) / tip(mesh, gf_ref_mild.components[0])
    print(f"mild_nu_ratio_gt_0p9={ratio_mild > 0.9}")
    if not (st_mild == 0 and ratio_mild > 0.9):
        print(f"FAIL: the same space also 'locks' at nu=1/3 "
              f"(ratio={ratio_mild:.4f}) -- effect is not incompressibility",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: order-2 pure displacement already recovers ----------
    st_p2, gf_p2 = solve_pure(mesh, NU_INCOMP, 2)
    ratio_p2 = tip(mesh, gf_p2) / ref
    print(f"order2_pure_within_1pct_of_mixed={abs(ratio_p2 - 1.0) < 0.01}")
    print(f"order2_pure_beats_order1={ratio_p2 > ratio_locked}")
    if st_p2 != 0 or abs(ratio_p2 - 1.0) >= 0.01:
        print(f"FAIL: order-2 pure displacement did not agree with mixed "
              f"(ratio={ratio_p2:.4f})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
