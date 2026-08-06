"""Tier-2: the J2 continuum tangent gives linear global Newton convergence; the
consistent (algorithmic) tangent gives quadratic.

Claim: ngsolve plasticity#6 -- "Consistent tangent modulus is required for
quadratic global Newton convergence; the continuum tangent (elastic + plastic
stiffness) does NOT match the linearization of the return-mapping discretization
and gives only linear convergence. ... Signal: with continuum tangent, Newton
residual norm drops by a factor of ~2-3 per iteration; with consistent tangent
the residual norm drops by ~residual^2 (quadratic)."

Wrong variant: a hand-rolled Newton whose tangent is the CONTINUUM
elastoplastic modulus assembled at the frozen current state,
    C_ep : de = C:de - 4*mu^2/(3*mu + H) * (n:de) n,   n = (3/2) s / q,
active only where q >= sigma_y, while the residual stays the exact return-map
residual.  The consistent tangent is what BilinearForm.AssembleLinearization
produces for that same residual form.

Setup: plane-strain J2 with linear isotropic hardening on unit_square maxh 0.3,
VectorH1 order 2, left clamped, right-face traction 400; E = 2e5, nu = 0.3,
sigma_y = 250, H = 2000; von Mises norm regularised by (1e-8*sigma_y)^2 so the
tangent is finite at u = 0.

Observed on NGSolve 6.2.2604:
  * in a fully elastic state the two tangent matrices are BITWISE identical
    (relative difference exactly 0.0) -- which is what validates the continuum
    implementation used here;
  * in a plastic state they differ by ~35 % in matrix norm;
  * consistent tangent: 9 residual evaluations, tail 1.27e-02 -> 5.54e-06 ->
    2.77e-12, i.e. quadratic;
  * continuum tangent: still not converged after 120 evaluations, with
    |r_{k+1}|/|r_k| pinned at a constant ~0.96.
So the qualitative claim holds exactly, but its stated rate is wrong in the
pessimistic direction: the residual does NOT drop "by a factor of ~2-3 per
iteration", it drops by about 4 % per iteration.  Both are pinned below.

Mutation control (re-runnable): T2_MUTATE=1 applies the documented fix at the
pathology site -- TANGENT_WRONG flips from "continuum" to "consistent", so the
second Newton run is driven by AssembleLinearization instead of the hand-built
C_ep.  It then converges in a handful of residuals and the fixture goes red on
continuum_hits_iteration_cap=True, continuum_still_above_tol=True,
continuum_ratio_constant_linear=True, continuum_drop_per_iter_faster_than_factor_2=False,
continuum_drop_per_iter_slower_than_4pct=True and consistent_needs_far_fewer_residuals=True.
Re-run: T2_MUTATE=1 python source.py
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
    Norm,
    Projector,
    Sym,
    Trace,
    VectorH1,
    ds,
    dx,
    sqrt,
    unit_square,
    x,
)

E, NU, SIG_Y, H_MOD = 200000.0, 0.3, 250.0, 2000.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
Q_REG = (1e-8 * SIG_Y) ** 2

TRACTION = 400.0
TOL = 1e-9
MAXIT_CONSISTENT = 30
MAXIT_CONTINUUM = 120

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The wrong variant drives Newton with the continuum tangent.
# T2_MUTATE=1 drives it with the consistent tangent instead.
TANGENT_WRONG = "consistent" if MUTATE else "continuum"

mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
fes = VectorH1(mesh, order=2, dirichlet="left")
u, v = fes.TnT()
PROJ_FREE = Projector(fes.FreeDofs(), True)


def embed(e2):
    return CoefficientFunction((e2[0, 0], e2[0, 1], 0.0,
                                e2[1, 0], e2[1, 1], 0.0,
                                0.0, 0.0, 0.0), dims=(3, 3))


def trial_state(uu):
    e3 = embed(Sym(Grad(uu)))
    sig_tr = 2 * MU * e3 + LAM * Trace(e3) * Id(3)
    s = Deviator(sig_tr)
    q = sqrt(1.5 * InnerProduct(s, s) + Q_REG)
    return sig_tr, s, q, q - SIG_Y


def residual_form(load: float):
    """Exact return-map residual; AssembleLinearization = consistent tangent."""
    sig_tr, s, q, f = trial_state(u)
    dgamma = IfPos(f, f / (3 * MU + H_MOD), 0.0)
    sig = sig_tr - 2 * MU * dgamma * 1.5 * s / q
    a = BilinearForm(fes)
    a += InnerProduct(sig, embed(Sym(Grad(v)))) * dx
    a += (-load * v[0]) * ds(definedon=mesh.Boundaries("right"))
    return a


def continuum_form(gfu):
    """C_ep : de = C:de - 4 mu^2/(3 mu + H) (n:de) n, state frozen at gfu."""
    _sig_tr, s, q, f = trial_state(gfu)
    n = 1.5 * s / q
    de = embed(Sym(Grad(u)))
    elastic = 2 * MU * de + LAM * Trace(de) * Id(3)
    plastic = (IfPos(f, 4 * MU * MU / (3 * MU + H_MOD), 0.0)
               * InnerProduct(n, de) * n)
    a = BilinearForm(fes)
    a += InnerProduct(elastic - plastic, embed(Sym(Grad(v)))) * dx
    return a


def mat_norm(mat) -> float:
    return float(np.linalg.norm(mat.AsVector().FV().NumPy()))


def rel_matrix_diff(gfu) -> tuple[float, str]:
    a = residual_form(0.0)
    a.AssembleLinearization(gfu.vec)
    ac = continuum_form(gfu)
    ac.Assemble()
    diff = a.mat.CreateMatrix()
    diff.AsVector().data = a.mat.AsVector() - ac.mat.AsVector()
    return mat_norm(diff) / mat_norm(a.mat), type(a.mat).__name__


def newton(load: float, tangent: str, maxit: int):
    a = residual_form(load)
    gfu = GridFunction(fes)
    r = gfu.vec.CreateVector()
    rf = gfu.vec.CreateVector()
    w = gfu.vec.CreateVector()
    hist = []
    for _it in range(maxit):
        a.Apply(gfu.vec, r)
        rf.data = PROJ_FREE * r
        hist.append(Norm(rf))
        if hist[-1] < TOL:
            break
        if tangent == "consistent":
            a.AssembleLinearization(gfu.vec)
            mat = a.mat
        else:
            ac = continuum_form(gfu)
            ac.Assemble()
            mat = ac.mat
        w.data = mat.Inverse(fes.FreeDofs()) * r
        gfu.vec.data -= w
    return hist, float(np.abs(gfu.components[0].vec.FV().NumPy()).max())


def main() -> int:
    ok = True

    # --- validation: in an elastic state the two tangents must coincide -----
    elastic_probe = GridFunction(fes)
    elastic_probe.Set(CoefficientFunction((1e-5 * x, 0.0)))
    d_el, mat_type = rel_matrix_diff(elastic_probe)
    print(f"tangent_matrix_type={mat_type} disp_space_type={type(fes).__name__}")
    print(f"elastic_state_tangents_identical={d_el == 0.0}")
    if d_el != 0.0:
        print(f"FAIL: the continuum tangent does not reduce to the elastic "
              f"stiffness below yield (rel={d_el:.3e})", file=sys.stderr)
        ok = False

    # --- and in a plastic state they must not -------------------------------
    plastic_probe = GridFunction(fes)
    plastic_probe.Set(CoefficientFunction((5e-3 * x, 0.0)))
    d_pl, _mt = rel_matrix_diff(plastic_probe)
    print(f"plastic_state_tangents_differ_gt_10pct={d_pl > 0.10}")
    if d_pl <= 0.10:
        print(f"FAIL: consistent and continuum tangents agree in the plastic "
              f"range (rel={d_pl:.3e})", file=sys.stderr)
        ok = False

    # --- RIGHT variant: consistent tangent ----------------------------------
    h_con, u_con = newton(TRACTION, "consistent", MAXIT_CONSISTENT)
    print(f"consistent_n_residuals={len(h_con)}")
    print(f"consistent_converged={h_con[-1] < TOL}")
    tail = [i for i in range(len(h_con) - 1) if 1e-11 < h_con[i] < 1.0]
    q_ratios = [h_con[i + 1] / h_con[i] ** 2 for i in tail]
    quadratic = len(q_ratios) >= 2 and max(q_ratios) / min(q_ratios) < 100.0
    print(f"consistent_quadratic={quadratic}")
    if not (h_con[-1] < TOL and quadratic):
        print(f"FAIL: the consistent tangent is not quadratic "
              f"(n={len(h_con)}, ratios={q_ratios})", file=sys.stderr)
        ok = False

    # --- WRONG variant: continuum tangent -----------------------------------
    h_cont, u_cont = newton(TRACTION, TANGENT_WRONG, MAXIT_CONTINUUM)
    print(f"continuum_n_residuals={len(h_cont)}")
    print(f"continuum_hits_iteration_cap={len(h_cont) == MAXIT_CONTINUUM}")
    print(f"continuum_still_above_tol={h_cont[-1] >= TOL}")
    ratios = [h_cont[i + 1] / h_cont[i] for i in range(len(h_cont) - 1)]
    tail_r = ratios[-5:]
    linear = all(0.5 < c < 0.999 for c in tail_r) and (max(tail_r) - min(tail_r)) < 0.01
    print(f"continuum_ratio_constant_linear={linear}")
    if not (len(h_cont) == MAXIT_CONTINUUM and linear):
        print(f"FAIL: the continuum tangent did not converge linearly "
              f"(n={len(h_cont)}, tail={tail_r})", file=sys.stderr)
        ok = False

    # --- the claimed rate "factor of ~2-3 per iteration" ---------------------
    print(f"continuum_drop_per_iter_faster_than_factor_2={max(tail_r) < 0.5}")
    print(f"continuum_drop_per_iter_slower_than_4pct={min(tail_r) > 0.90}")
    if max(tail_r) < 0.5:
        print("FAIL: the continuum tangent now drops by a factor >= 2 per "
              "iteration -- revisit the rate falsification recorded here",
              file=sys.stderr)
        ok = False

    # --- both are heading for the same root ---------------------------------
    print(f"continuum_within_1pct_of_consistent_solution="
          f"{abs(u_cont - u_con) < 0.01 * abs(u_con)}")
    print(f"consistent_needs_far_fewer_residuals={len(h_con) * 5 < len(h_cont)}")
    if abs(u_cont - u_con) >= 0.01 * abs(u_con):
        print(f"FAIL: the two tangents are converging to different solutions "
              f"({u_cont} vs {u_con})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
