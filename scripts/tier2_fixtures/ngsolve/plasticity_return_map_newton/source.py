"""Tier-2: the per-integration-point return map converges quadratically in an
iteration count that does not depend on the mesh -- and NewtonCF signals an
under-iterated local solve by returning NaN.

Claim: ngsolve plasticity#2 -- the return mapping is the integration-point step
(elastic trial stress, yield check, radial return along the trial deviator).
"With NewtonCF this becomes a per-integration-point Newton solve.  Signal:
iteration count to drive |sigma_vm - (sigma_y + H*alpha)| below 1e-10 is bounded
independently of mesh size for quadratic Newton convergence."

Wrong variant: run the same local problem through ngsolve.fem.NewtonCF with
maxiter below what the local Newton needs.

Setup: one analytic non-uniform strain field on VectorH1 order 2, power-law
isotropic hardening Y(a) = sigma_y + K*a^n (n = 0.4) so the local problem is
genuinely nonlinear, residual r(dg) = q_trial - 3*mu*dg - Y(alpha + dg), and the
local unknown carried on an L2 order-0 trial function.

Observed on NGSolve 6.2.2604, meshes of 14 / 54 / 224 elements:
  * the smallest maxiter that returns a non-NaN result is 7 on ALL THREE meshes;
  * with maxiter below that, NewtonCF returns NaN -- its documented failure
    signal;
  * the residual history is 1.86e+01 -> 6.14e-03 -> 8.25e-10 -> ~1e-13, i.e.
    r_{k+1}/r_k^2 pinned near 2e-5: quadratic;
  * after convergence the area fraction where |r| > 1e-10 is exactly 0.0, on
    every mesh -- the claim's stated criterion, met.

Mutation control (re-runnable): T2_MUTATE=1 applies the documented fix at the
pathology site -- MAXITER_WRONG goes from 4 to 20, i.e. the local NewtonCF solve
is given enough iterations to converge.  It no longer returns NaN and the fixture
goes red on returns_nan=True.  Re-run: T2_MUTATE=1 python source.py
"""

from __future__ import annotations

import math
import os
import sys

from ngsolve import (
    CoefficientFunction,
    Deviator,
    GridFunction,
    Grad,
    Id,
    IfPos,
    InnerProduct,
    Integrate,
    L2,
    Mesh,
    Sym,
    Trace,
    VectorH1,
    sqrt,
    unit_square,
    x,
    y,
)
from ngsolve.fem import NewtonCF

E, NU, SIG_Y = 200000.0, 0.3, 250.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
K_HARD, N_HARD = 3000.0, 0.4      # power-law hardening -> nonlinear local problem

MESHES = (0.4, 0.2, 0.1)

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The wrong variant under-iterates the local Newton.
# T2_MUTATE=1 gives it enough iterations instead.
MAXITER_WRONG = 20 if MUTATE else 4


def hardening(a):
    return SIG_Y + K_HARD * (a + 1e-12) ** N_HARD


def setup(maxh: float):
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    area = Integrate(CoefficientFunction(1.0), mesh)
    fes = VectorH1(mesh, order=2)
    gfu = GridFunction(fes)
    gfu.Set(CoefficientFunction((0.010 * x * (1 + 0.5 * y), -0.003 * y)))
    e2 = Sym(Grad(gfu))
    e3 = CoefficientFunction((e2[0, 0], e2[0, 1], 0.0,
                              e2[1, 0], e2[1, 1], 0.0,
                              0.0, 0.0, 0.0), dims=(3, 3))
    sig_tr = 2 * MU * e3 + LAM * Trace(e3) * Id(3)
    q_tr = sqrt(1.5 * InnerProduct(Deviator(sig_tr), Deviator(sig_tr)))
    local = L2(mesh, order=0)
    start = GridFunction(local)
    start.vec[:] = 0.0
    return mesh, area, q_tr, local.TrialFunction(), start


def main() -> int:
    ok = True

    doc = NewtonCF.__doc__ or ""
    nan_doc = "convergence failure is indicated by returning NaNs" in doc
    print(f"newtoncf_module={NewtonCF.__module__}")
    print(f"newtoncf_doc_says_nan_on_failure={nan_doc}")
    if not nan_doc:
        print("FAIL: NewtonCF no longer documents NaN as the failure signal",
              file=sys.stderr)
        ok = False

    min_iters = []
    hist_all = []
    for maxh in MESHES:
        mesh, area, q_tr, dg_trial, start = setup(maxh)
        res = q_tr - 3 * MU * dg_trial - hardening(dg_trial)

        yielding = Integrate(IfPos(q_tr - SIG_Y, 1.0, 0.0), mesh) / area
        if maxh == MESHES[0]:
            print(f"trial_state_is_plastic_everywhere={abs(yielding - 1.0) < 1e-9}")
            if abs(yielding - 1.0) >= 1e-9:
                print(f"FAIL: the trial state is not beyond yield "
                      f"(fraction={yielding})", file=sys.stderr)
                ok = False

        # --- smallest maxiter that does not return NaN ---------------------
        kmin = None
        for k in range(1, 16):
            val = Integrate(NewtonCF(res, start, tol=1e-12, maxiter=k), mesh) / area
            if not math.isnan(val):
                kmin = k
                break
        min_iters.append(kmin)

        # --- residual history: allow_fail returns the k-th iterate ----------
        hist = []
        for k in range(1, 9):
            dg_k = NewtonCF(res, start, tol=1e-30, maxiter=k, allow_fail=True)
            r_k = q_tr - 3 * MU * dg_k - hardening(dg_k)
            hist.append(Integrate(sqrt(r_k * r_k), mesh) / area)
        hist_all.append(hist)

        # --- the claim's own criterion, after convergence -------------------
        dg = NewtonCF(res, start, tol=1e-12, maxiter=20)
        r = q_tr - 3 * MU * dg - hardening(dg)
        frac_bad = Integrate(IfPos(sqrt(r * r) - 1e-10, 1.0, 0.0), mesh) / area
        print(f"maxh{maxh}_ne={mesh.ne} "
              f"consistency_below_1e-10_everywhere={frac_bad == 0.0}")
        if frac_bad != 0.0:
            print(f"FAIL: |sigma_vm - Y| stayed above 1e-10 on {frac_bad:.3f} of "
                  f"the domain at maxh={maxh}", file=sys.stderr)
            ok = False

    # --- mesh independence --------------------------------------------------
    print(f"min_maxiter_per_mesh={min_iters}")
    same = len(set(min_iters)) == 1 and min_iters[0] is not None
    print(f"min_maxiter_identical_across_meshes={same}")
    print(f"min_maxiter_bounded_le_10={all(k is not None and k <= 10 for k in min_iters)}")
    if not same:
        print(f"FAIL: local iteration count depends on the mesh "
              f"({min_iters})", file=sys.stderr)
        ok = False

    # --- quadratic convergence of the local Newton --------------------------
    hist = hist_all[0]
    tail = [i for i in range(len(hist) - 1) if 1e-9 < hist[i] < 1e2]
    ratios = [hist[i + 1] / hist[i] ** 2 for i in tail]
    quadratic = len(ratios) >= 2 and max(ratios) / min(ratios) < 10.0
    print(f"local_newton_quadratic={quadratic}")
    print(f"local_residual_reaches_machine_precision={min(hist) < 1e-12}")
    if not quadratic:
        print(f"FAIL: the local Newton is not quadratic "
              f"(ratios={ratios})", file=sys.stderr)
        ok = False

    # --- WRONG variant: under-iterated local solve --------------------------
    mesh, area, q_tr, dg_trial, start = setup(MESHES[0])
    res = q_tr - 3 * MU * dg_trial - hardening(dg_trial)
    bad = Integrate(NewtonCF(res, start, tol=1e-12, maxiter=MAXITER_WRONG),
                    mesh) / area
    print(f"underiterated_maxiter={MAXITER_WRONG} returns_nan={math.isnan(bad)}")
    if not math.isnan(bad):
        print(f"FAIL: NewtonCF with maxiter={MAXITER_WRONG} converged "
              f"(value={bad}) -- the pitfall did not reproduce", file=sys.stderr)
        ok = False

    # --- and the pitfall the return map exists to avoid ---------------------
    elastic_violation = Integrate(IfPos(q_tr - hardening(0.0), 1.0, 0.0),
                                  mesh) / area
    print(f"elastic_trial_violates_yield_everywhere="
          f"{abs(elastic_violation - 1.0) < 1e-9}")
    if abs(elastic_violation - 1.0) >= 1e-9:
        print("FAIL: the untreated elastic trial stress does not violate the "
              "yield surface", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
