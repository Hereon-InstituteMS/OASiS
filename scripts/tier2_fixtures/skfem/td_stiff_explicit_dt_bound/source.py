"""Tier-2: the explicit dt bound on a stiff reaction-diffusion system.

Claim: skfem time_dependent#5 -- for stiff (reaction-dominated) systems prefer
backward Euler or BDF2: "explicit (theta=0) or near-explicit (theta<0.5) on
stiff problems requires dt < 2/lambda_max, where lambda_max is the largest
eigenvalue of M^-1 K (with M from BilinearForm asm of mass form); for
reaction-diffusion with Da > 100, characteristic dt is so small the simulation
is infeasible."

Measured on skfem 12.0.1, u_t = Laplace u - Da*u on a 12x12
MeshTri.init_tensor with ElementTriP1, homogeneous Dirichlet, lambda_max taken
from scipy.linalg.eigh(K_free, M_free):

  * THE BOUND IS SHARP AND IT IS 2/lambda_max, not a rule of thumb.  At
    dt = 0.9 * 2/lambda_max the explicit run decays over hundreds of steps;
    at dt = 1.1 * 2/lambda_max it grows geometrically past 1e26 in the same
    number of steps.  Both sides were measured, so the fixture pins the
    transition rather than only one side of it.
  * ADDING REACTION SHIFTS lambda_max BY EXACTLY Da.  With the reaction term
    -Da*u the operator is K + Da*M, so lambda_max(M^-1(K + Da*M)) is
    lambda_max(M^-1 K) + Da to machine precision -- which is why a large Da
    tightens the step bound directly.
  * THE "INFEASIBLE" PART DEPENDS ON THE MESH, NOT ON Da ALONE.  On this
    mesh the diffusive lambda_max already dwarfs Da = 100, so Da = 100 barely
    moves the bound; Da has to reach the same order as lambda_max(M^-1 K)
    before it dominates.  Da > 100 is not by itself a stiffness threshold.
  * BACKWARD EULER AT THE SAME dt IS FINE, which is the recommendation.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import scipy.linalg as sla
from skfem import Basis, BilinearForm, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

NX = 12


@BilinearForm
def mass_form(u, v, w):
    return u * v


def build(da):
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    M = mass_form.assemble(ib)
    D = ib.get_dofs().all()
    return ib, (K + da * M).tocsr(), M.tocsr(), D


def lambda_max(op, M, D, N):
    free = np.setdiff1d(np.arange(N), D)
    ev = sla.eigh(op[free][:, free].toarray(), M[free][:, free].toarray(),
                  eigvals_only=True)
    return float(ev.max())


def explicit_run(da, dt, nsteps=400):
    ib, op, M, D = build(da)
    u = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u[D] = 0.0
    peak = [float(np.abs(u).max())]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(nsteps):
            rhs = M @ u - dt * (op @ u)
            u = solve(*condense(M, rhs, D=D))
            peak.append(float(np.abs(u).max()))
            if not np.isfinite(peak[-1]) or peak[-1] > 1e30:
                break
        msgs = sorted({str(c.message) for c in caught})
    return peak, msgs


def implicit_run(da, dt, nsteps=400):
    ib, op, M, D = build(da)
    u = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u[D] = 0.0
    A = (M + dt * op).tocsr()
    peak = [float(np.abs(u).max())]
    for _ in range(nsteps):
        u = solve(*condense(A, M @ u, D=D))
        peak.append(float(np.abs(u).max()))
    return peak


def main() -> int:
    ok = True
    ib, K, M, D = build(0.0)
    lam_diff = lambda_max(K, M, D, ib.N)
    print(f"dofs_N={ib.N}")
    print(f"lambda_max_diffusion_only={lam_diff:.6e}")

    # --- the reaction term shifts the spectrum by exactly Da -------------
    for da in (100.0, 10000.0):
        _, op, Mda, Dda = build(da)
        lam = lambda_max(op, Mda, Dda, ib.N)
        shift = lam - lam_diff
        print(f"da{int(da)}_lambda_max={lam:.6e}")
        print(f"da{int(da)}_shift_minus_da={shift - da:.3e}")
        print(f"da{int(da)}_shift_is_exactly_da={abs(shift - da) < 1e-6}")
        print(f"da{int(da)}_dominates_diffusion={da > lam_diff}")
        if abs(shift - da) >= 1e-6:
            print(f"FAIL: adding Da = {da} did not shift lambda_max by Da",
                  file=sys.stderr)
            ok = False
    print(f"da_100_is_a_stiffness_threshold_on_this_mesh="
          f"{100.0 > lam_diff}")

    # --- the bound is sharp ---------------------------------------------
    da = 100.0
    _, op, Mda, Dda = build(da)
    lam = lambda_max(op, Mda, Dda, ib.N)
    dt_crit = 2.0 / lam
    print(f"dt_crit_2_over_lambda_max={dt_crit:.6e}")
    for factor, tag in ((0.9, "below"), (1.1, "above")):
        peak, msgs = explicit_run(da, factor * dt_crit)
        # blow-up means the amplitude GREW away from the initial state, not
        # that it happened to reach a particular float; the diffusive
        # solution must decay
        grew = (not np.isfinite(peak[-1])) or peak[-1] > 1e6 * peak[0]
        print(f"explicit_{tag}_bound_initial_peak={peak[0]:.4e}")
        print(f"explicit_{tag}_bound_final_peak={peak[-1]:.4e}")
        print(f"explicit_{tag}_bound_blew_up={grew}")
        print(f"explicit_{tag}_bound_steps_taken={len(peak) - 1}")
        print(f"explicit_{tag}_bound_warnings={msgs!r}")
        if tag == "below" and grew:
            print("FAIL: the explicit run blew up BELOW 2/lambda_max",
                  file=sys.stderr)
            ok = False
        if tag == "above" and not grew:
            print("FAIL: the explicit run stayed bounded ABOVE 2/lambda_max",
                  file=sys.stderr)
            ok = False

    # --- backward Euler at the unstable dt is fine ----------------------
    peak_imp = implicit_run(da, 1.1 * dt_crit)
    print(f"implicit_same_dt_final_peak={peak_imp[-1]:.4e}")
    print(f"implicit_same_dt_stays_bounded="
          f"{bool(np.isfinite(peak_imp[-1]) and peak_imp[-1] <= peak_imp[0])}")
    print(f"implicit_decays_monotonically="
          f"{all(peak_imp[i + 1] <= peak_imp[i] + 1e-12
                 for i in range(len(peak_imp) - 1))}")
    if not (np.isfinite(peak_imp[-1]) and peak_imp[-1] <= peak_imp[0]):
        print("FAIL: backward Euler did not survive the unstable dt",
              file=sys.stderr)
        ok = False

    # --- how small does dt have to be at a truly stiff Da? --------------
    big = 1e5
    _, op_big, M_big, D_big = build(big)
    lam_big = lambda_max(op_big, M_big, D_big, ib.N)
    print(f"stiff_da={int(big)} stiff_lambda_max={lam_big:.4e}")
    print(f"stiff_dt_crit={2.0 / lam_big:.4e}")
    print(f"stiff_dt_crit_smaller_than_mild="
          f"{2.0 / lam_big < dt_crit}")
    print(f"steps_for_unit_time_at_stiff_da="
          f"{int(np.ceil(1.0 / (2.0 / lam_big)))}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
