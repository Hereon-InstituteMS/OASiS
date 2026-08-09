"""Tier-2: the real divergence signature of solvers.Newton on a J2 load step is
(-1, numit) -- the claim has the tuple the other way round.

Claim: ngsolve plasticity#4 -- load stepping is required, not optional, for J2 +
hardening.  "Signal: with too-large step size, solvers.Newton called on the
per-step BilinearForm returns (iters, conv) with iters == maxit and conv > maxerr
-- the divergence signature.  Reduce the load increment (more n_load_steps), not
the maxit kwarg of Newton -- increasing maxit just delays the same convergence
failure."

Wrong variant: apply the whole right-face traction in ONE step.

Setup: plane-strain J2 with linear isotropic hardening written as
CoefficientFunction algebra (Sym, Deviator, IfPos) on unit_square maxh 0.3,
VectorH1 order 2, left face clamped, traction on the right face; the von Mises
norm is regularised as sqrt(1.5*s:s + (1e-8*sigma_y)^2) so that the
AssembleLinearization tangent is finite at u = 0.  E = 2e5, nu = 0.3,
sigma_y = 250, H = 100, total traction 500.

Observed on NGSolve 6.2.2604:
  * one shot -> solvers.Newton returns (-1, 30) with maxit=30, (-1, 100) with
    maxit=100, (-1, 200) with maxit=200, printing
    "Warning: Newton might not converge!  Error =  1.9815674139..." every time.
    So the guidance "increase the load steps, not maxit" is CONFIRMED, and
    strongly: the stalled error is identical to 10 digits at every maxit.
  * the returned pair is (status, numit): the FIRST element is the -1/0
    convergence flag and the SECOND is the iteration count that equals maxit.
    The claim names them (iters, conv) in the opposite order and expects an
    error value that solvers.Newton never returns.  This is the same misread
    that makes the shipped hyperelasticity templates print "Newton iters=0".
  * per-step increment 100 % and 40 % of sigma_y diverge; 20 %, 10 % and 5 %
    all converge and land on the same max|u_x| to 6 digits.

Mutation control (re-runnable): T2_MUTATE=1 applies the documented fix at the
pathology site -- N_STEPS_WRONG goes from 1 to 10, i.e. the traction is applied
in 20 %-of-sigma_y increments instead of one shot, which is the guidance's own
remedy ("reduce the load increment, not maxit").  solvers.Newton then returns
status 0 and the fixture goes red on oneshot_first_element=-1,
oneshot_diverged_at_every_maxit=True, oneshot_second_element_equals_maxit=True and
raising_maxit_does_not_help=True.  Re-run: T2_MUTATE=1 python source.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    Deviator,
    GridFunction,
    Grad,
    Id,
    IfPos,
    InnerProduct,
    Mesh,
    Sym,
    Trace,
    VectorH1,
    ds,
    dx,
    solvers,
    sqrt,
    unit_square,
)

E, NU, SIG_Y, H_MOD = 200000.0, 0.3, 250.0, 100.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
Q_REG = (1e-8 * SIG_Y) ** 2

TOTAL_TRACTION = 500.0

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The wrong variant applies the whole traction in one step.
# T2_MUTATE=1 applies the documented remedy: smaller load increments.
N_STEPS_WRONG = 10 if MUTATE else 1

mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
fes = VectorH1(mesh, order=2, dirichlet="left")
u, v = fes.TnT()


def embed(e2):
    return CoefficientFunction((e2[0, 0], e2[0, 1], 0.0,
                                e2[1, 0], e2[1, 1], 0.0,
                                0.0, 0.0, 0.0), dims=(3, 3))


def j2_stress(uu):
    e3 = embed(Sym(Grad(uu)))
    sig_tr = 2 * MU * e3 + LAM * Trace(e3) * Id(3)
    s = Deviator(sig_tr)
    q = sqrt(1.5 * InnerProduct(s, s) + Q_REG)
    f = q - SIG_Y
    dgamma = IfPos(f, f / (3 * MU + H_MOD), 0.0)
    return sig_tr - 2 * MU * dgamma * 1.5 * s / q


def form(load: float):
    a = BilinearForm(fes)
    a += InnerProduct(j2_stress(u), embed(Sym(Grad(v)))) * dx
    a += (-load * v[0]) * ds(definedon=mesh.Boundaries("right"))
    return a


def run(n_steps: int, maxit: int = 30):
    gfu = GridFunction(fes)
    returns = []
    for k in range(1, n_steps + 1):
        returns.append(solvers.Newton(form(k / n_steps * TOTAL_TRACTION), gfu,
                                      maxit=maxit, printing=False, maxerr=1e-8))
    u_max = float(np.abs(gfu.components[0].vec.FV().NumPy()).max())
    return returns, u_max


def main() -> int:
    ok = True

    doc = solvers.Newton.__doc__ or ""
    doc_ok = "0 if Newton's method did converge" in doc
    print(f"newton_doc_first_return_is_status={doc_ok}")
    if not doc_ok:
        print("FAIL: solvers.Newton no longer documents (status, numit)",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: full traction in one step --------------------------
    per_maxit = {}
    for maxit in (30, 100, 200):
        rets, _u = run(N_STEPS_WRONG, maxit)
        per_maxit[maxit] = rets[-1]
    print(f"oneshot_returns={per_maxit}")
    first, second = per_maxit[30]
    diverged = all(r[0] == -1 for r in per_maxit.values())
    print(f"oneshot_first_element={first}")
    print(f"oneshot_diverged_at_every_maxit={diverged}")
    print("oneshot_second_element_equals_maxit="
          f"{all(per_maxit[m][1] == m for m in per_maxit)}")
    print(f"oneshot_first_element_equals_maxit={first == 30}")
    if not diverged:
        print(f"FAIL: the one-shot traction converged ({per_maxit}) -- the "
              f"pitfall did not reproduce", file=sys.stderr)
        ok = False
    if first == 30:
        print("FAIL: the first return element now equals maxit -- revisit the "
              "tuple-order falsification recorded here", file=sys.stderr)
        ok = False
    print(f"oneshot_second_element_is_iteration_count={second == 30}")

    # --- raising maxit is not the fix --------------------------------------
    print("raising_maxit_does_not_help="
          f"{all(per_maxit[m][0] == -1 for m in (100, 200))}")

    # --- RIGHT variant: reduce the increment -------------------------------
    results = {}
    for n in (2, 5, 10, 20, 40):
        rets, u_max = run(n)
        results[n] = (all(r[0] == 0 for r in rets), u_max,
                      TOTAL_TRACTION / n / SIG_Y)
    for n, (conv, u_max, frac) in results.items():
        print(f"n_steps={n} increment_pct_of_sigma_y={frac * 100:.0f} "
              f"all_steps_converged={conv}")
    coarse_fails = not results[2][0] and not results[5][0]
    fine_works = all(results[n][0] for n in (10, 20, 40))
    print(f"coarse_increments_diverge={coarse_fails}")
    print(f"increment_at_or_below_10pct_of_sigma_y_converges="
          f"{results[20][0] and results[40][0]}")
    print(f"all_fine_step_sizes_converge={fine_works}")
    if not (coarse_fails and fine_works):
        print(f"FAIL: load stepping did not separate coarse from fine "
              f"({[(n, r[0]) for n, r in results.items()]})", file=sys.stderr)
        ok = False

    agree = max(abs(results[n][1] - results[10][1]) for n in (10, 20, 40))
    print(f"converged_solution_independent_of_step_size={agree < 1e-6}")
    if agree >= 1e-6:
        print(f"FAIL: converged solutions disagree across step sizes "
              f"(spread={agree:.3e})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
