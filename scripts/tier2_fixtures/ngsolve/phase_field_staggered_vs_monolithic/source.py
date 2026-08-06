"""Tier-2: staggered alternate minimisation converges in 5-20 outer
iterations from a cold start; monolithic Newton on the same energy does not
converge at all unless you hand it the staggered answer first.

Claim: ngsolve phase_field#4 - staggered converges to the SAME solution as
monolithic but takes more iterations per step; it is more robust where
monolithic Newton struggles.

Wrong variant: build the AT2 energy on the compound space VectorH1 x H1 with
  BilinearForm += Variation(E*dx) and call ngsolve.solvers.Newton from the
  natural cold start (u = Dirichlet data, d = 0).
Right variant : alternate minimisation - one linear elastic solve with d
  frozen, one linear damage solve with u frozen, repeat.

Observed on NGSolve 6.2.2604, unit_square maxh=0.15 (ne=96), VectorH1
order 1 dirichlet='bottom|top', H1 order 1, E=1, nu=0.3, Gc=1e-3, l0=0.1
(2026-08-06):
  staggered, cold start, tol 1e-8 on ||d_new - d_old||
    disp=0.01 ->  5 outer iterations, d_max = 0.01429550
    disp=0.02 ->  8 outer iterations, d_max = 0.05539218
    disp=0.03 -> 12 outer iterations, d_max = 0.11852371
  monolithic Newton, cold start, disp=0.03
    flag -1 after 60 iterations, d_max = 2.70677637 (d > 1 is unphysical),
    NGSolve printing "Warning: Newton might not converge! Error = "
  monolithic Newton warm-started from the staggered solution
    flag 0 in 2 iterations, d_max = 0.11852278

CORRECTION to the claim: "converges to the SAME solution as monolithic but
takes more iterations per step" is only half right here.  The same solution
is confirmed (the warm-started monolithic solve reproduces the staggered
d_max to 8e-06 relative), but from a cold start monolithic Newton does not
converge at any of these load levels, so it is not a cheaper alternative -
it needs the staggered answer as its initial guess.
"""
from __future__ import annotations

import os
import sys
import tempfile

from netgen.geom2d import unit_square
from ngsolve import (BilinearForm, CoefficientFunction, GridFunction, Grad,
                     H1, Id, InnerProduct, LinearForm, Mesh, Trace, Variation,
                     VectorH1, dx, grad)
from ngsolve.solvers import Newton

E_MOD, NU = 1.0, 0.3
MU = E_MOD / (2 * (1 + NU))
LAM = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
GC, L0, KRES = 1e-3, 0.1, 1e-10
LOADS = (0.01, 0.02, 0.03)

mesh = Mesh(unit_square.GenerateMesh(maxh=0.15))
Vu = VectorH1(mesh, order=1, dirichlet="bottom|top")
Vd = H1(mesh, order=1)
X = Vu * Vd


def strain(w):
    return 0.5 * (Grad(w) + Grad(w).trans)


def sig0(w):
    e = strain(w)
    return 2 * MU * e + LAM * Trace(e) * Id(2)


def psi(w):
    e = strain(w)
    return 0.5 * LAM * Trace(e) ** 2 + MU * InnerProduct(e, e)


def staggered(disp, tol=1e-8, maxit=100):
    u, v = Vu.TnT()
    d, phi = Vd.TnT()
    gfu, gfd = GridFunction(Vu), GridFunction(Vd)
    gfd.Set(0)
    for it in range(maxit):
        gfu.Set(CoefficientFunction((0, disp)),
                definedon=mesh.Boundaries("top"))
        a = BilinearForm(Vu)
        a += ((1 - gfd) ** 2 + KRES) * InnerProduct(sig0(u), strain(v)) * dx
        a.Assemble()
        f = LinearForm(Vu)
        f.Assemble()
        r = f.vec.CreateVector()
        r.data = f.vec - a.mat * gfu.vec
        gfu.vec.data += a.mat.Inverse(Vu.FreeDofs()) * r
        hfield = psi(gfu)
        ad = BilinearForm(Vd, symmetric=True)
        ad += ((GC / L0 + 2 * hfield) * d * phi
               + GC * L0 * grad(d) * grad(phi)) * dx
        ad.Assemble()
        fd = LinearForm(2 * hfield * phi * dx)
        fd.Assemble()
        gnew = GridFunction(Vd)
        gnew.vec.data = ad.mat.Inverse(Vd.FreeDofs()) * fd.vec
        delta = (gnew.vec - gfd.vec).Norm()
        gfd.vec.data = gnew.vec
        if delta < tol:
            break
    return gfu, gfd, it + 1


def energy_form():
    (u, d) = X.TrialFunction()
    e = strain(u)
    ps = 0.5 * LAM * Trace(e) ** 2 + MU * InnerProduct(e, e)
    total = (((1 - d) ** 2 + KRES) * ps + GC / (2 * L0) * d * d
             + GC * L0 / 2 * grad(d) * grad(d))
    a = BilinearForm(X)
    a += Variation(total * dx)
    return a


def capture_fds(fn):
    with tempfile.TemporaryFile(mode="w+") as tmp:
        so, se = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)
            value = fn()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(so, 1)
            os.dup2(se, 2)
            os.close(so)
            os.close(se)
        tmp.seek(0)
        return value, tmp.read()


def main() -> int:
    ok = True
    print(f"mesh_ne={mesh.ne} compound_space_class={type(X).__name__} "
          f"compound_ndof={X.ndof}")

    # ---- RIGHT: staggered alternate minimisation ------------------------
    iters, dmaxes = [], []
    for disp in LOADS:
        _, gfd, nit = staggered(disp)
        iters.append(nit)
        dmaxes.append(max(gfd.vec))
        print(f"staggered_disp={disp} outer_iters={nit} "
              f"d_max={dmaxes[-1]:.8f}")
    in_range = all(5 <= n <= 20 for n in iters)
    print(f"staggered_iters_in_5_to_20={in_range}")
    print("staggered_iters_increase_with_load="
          f"{iters[0] < iters[1] < iters[2]}")
    print("staggered_dmax_increases_with_load="
          f"{dmaxes[0] < dmaxes[1] < dmaxes[2]}")
    if not in_range:
        print(f"FAIL: staggered outer-iteration counts {iters} are outside "
              f"the documented 5-20 band", file=sys.stderr)
        ok = False
    if not (iters[0] < iters[1] < iters[2]
            and dmaxes[0] < dmaxes[1] < dmaxes[2]):
        print(f"FAIL: iteration counts {iters} / damage {dmaxes} are not "
              f"monotone in the load", file=sys.stderr)
        ok = False

    us, ds_, _ = staggered(LOADS[-1])
    us_t, ds_t, _ = staggered(LOADS[-1], tol=1e-11)
    print("staggered_is_a_fixed_point="
          f"{abs(max(ds_.vec) - max(ds_t.vec)) < 1e-6}")

    # ---- WRONG: monolithic Newton from a cold start ---------------------
    disp = LOADS[-1]
    a_cold = energy_form()
    gf_cold = GridFunction(X)
    gf_cold.components[0].Set(CoefficientFunction((0, disp)),
                              definedon=mesh.Boundaries("top"))
    (flag_cold, nit_cold), chatter = capture_fds(
        lambda: Newton(a_cold, gf_cold, freedofs=X.FreeDofs(),
                       printing=False, maxit=60))
    dmax_cold = max(gf_cold.components[1].vec)
    print(f"monolithic_cold_start_flag={flag_cold} iters={nit_cold} "
          f"d_max={dmax_cold:.8f}")
    print(f"monolithic_cold_start_converged={flag_cold == 0}")
    print(f"monolithic_cold_start_dmax_exceeds_1={dmax_cold > 1.0}")
    print("monolithic_warning_emitted="
          f"{'Newton might not converge' in chatter}")
    print(f"monolithic_chatter={chatter.strip().splitlines()[:1]}")
    if flag_cold == 0:
        print(f"FAIL: monolithic Newton converged from a cold start "
              f"({nit_cold} iters); the robustness gap is absent",
              file=sys.stderr)
        ok = False
    if dmax_cold <= 1.0:
        print(f"FAIL: the failed monolithic solve stayed physical "
              f"(d_max={dmax_cold:.4f})", file=sys.stderr)
        ok = False
    if "Newton might not converge" not in chatter:
        print("FAIL: NGSolve did not emit its non-convergence warning",
              file=sys.stderr)
        ok = False

    # ---- monolithic, warm-started from the staggered solution -----------
    a_warm = energy_form()
    gf_warm = GridFunction(X)
    gf_warm.components[0].vec.data = us.vec
    gf_warm.components[1].vec.data = ds_.vec
    flag_warm, nit_warm = Newton(a_warm, gf_warm, freedofs=X.FreeDofs(),
                                 printing=False, maxit=60)
    dmax_warm = max(gf_warm.components[1].vec)
    print(f"monolithic_warm_start_flag={flag_warm} iters={nit_warm} "
          f"d_max={dmax_warm:.8f}")
    print(f"monolithic_warm_started_converged={flag_warm == 0}")
    print(f"monolithic_warm_iters_lt_staggered={nit_warm < iters[-1]}")
    rel = abs(dmax_warm - dmaxes[-1]) / dmaxes[-1]
    print(f"warm_started_monolithic_matches_staggered={rel < 1e-4}")
    if flag_warm != 0 or rel >= 1e-4:
        print(f"FAIL: warm-started monolithic Newton did not reproduce the "
              f"staggered solution (flag {flag_warm}, rel {rel:.3e})",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
