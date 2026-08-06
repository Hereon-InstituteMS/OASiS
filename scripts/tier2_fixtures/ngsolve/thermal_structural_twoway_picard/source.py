"""Tier-2: one-way thermal->structural is wrong by 5-20% when k depends on u.

Claim: ngsolve thermal_structural#3 — for two-way coupling, iterate
thermal -> structural -> thermal until both GridFunction fields converge
(typically 3-10 Picard iterations). Doing only the FIRST thermal solve and the
FIRST structural solve gives a deformation that never feeds back into the
conductivity CoefficientFunction; for problems where deformation changes the
effective k(u), the one-way result is wrong by 5-20% on the temperature
distribution. Track ||T_new - T_old||/||T_new|| and ||u_new - u_old||/||u_new||
< 1e-4 to stop.

Wrong variant: stop after the first pair of solves, i.e. a single sweep.

Problem: unit_square maxh=0.3, T = 100 on the left edge and 0 on the right,
k(u) = k0 * (1 + 50 * Trace(Strain(u))), structural part clamped on the left
with alpha = 2.0e-4, E = 210 GPa, nu = 0.3. The deformation is non-uniform
because T is, so k(u) varies in space and genuinely reshapes T.

Observed on NGSolve 6.2.2604 (2026-08-03):
  * the Picard loop reaches ||dT||/||T|| and ||du||/||u|| < 1e-4 in 6 sweeps;
  * the one-way temperature field is 13.89% away from the converged one in
    relative L2 -- inside the documented 5-20% band;
  * the residual history is monotonically decreasing.
"""
from __future__ import annotations

import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: how many thermal->structural sweeps are taken (1 == one-way)
N_PICARD_SWEEPS = 1

MAX_SWEEPS = 20
TOL = 1e-4
E, NU, ALPHA, DT = 210e3, 0.3, 2.0e-4, 100.0
MU = E / (2.0 * (1.0 + NU))
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
K0, C_K = 1.0, 50.0                    # k(u) = K0 * (1 + C_K * div u)

mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.3))
V_T = ngs.H1(mesh, order=2, dirichlet="left|right")
V_u = ngs.VectorH1(mesh, order=2, dirichlet="left")
uT, vT = V_T.TnT()
uu, vv = V_u.TnT()


def strain(w):
    return 0.5 * (ngs.Grad(w) + ngs.Grad(w).trans)


def thermal_solve(gfu) -> ngs.GridFunction:
    k = K0 * (1.0 + C_K * ngs.Trace(strain(gfu)))
    a = ngs.BilinearForm(k * ngs.grad(uT) * ngs.grad(vT) * ngs.dx).Assemble()
    f = ngs.LinearForm(V_T)
    f.Assemble()
    gfT = ngs.GridFunction(V_T)
    gfT.Set(ngs.IfPos(0.5 - ngs.x, DT, 0.0),
            definedon=mesh.Boundaries("left|right"))
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gfT.vec
    gfT.vec.data += a.mat.Inverse(V_T.FreeDofs()) * r
    return gfT


def structural_solve(gfT) -> ngs.GridFunction:
    a = ngs.BilinearForm(ngs.InnerProduct(
        2 * MU * strain(uu) + LAM * ngs.Trace(strain(uu)) * ngs.Id(2),
        strain(vv)) * ngs.dx).Assemble()
    f = ngs.LinearForm(ngs.InnerProduct(
        (3 * LAM + 2 * MU) * ALPHA * gfT * ngs.Id(2), strain(vv)) * ngs.dx
    ).Assemble()
    gfu = ngs.GridFunction(V_u)
    gfu.vec.data = a.mat.Inverse(V_u.FreeDofs()) * f.vec
    return gfu


def rel_l2(new, old, vector: bool) -> float:
    if vector:
        num = ngs.Integrate(ngs.InnerProduct(new - old, new - old), mesh)
        den = ngs.Integrate(ngs.InnerProduct(new, new), mesh)
    else:
        num = ngs.Integrate((new - old) ** 2, mesh)
        den = ngs.Integrate(new ** 2, mesh)
    return float(abs(num) ** 0.5 / abs(den) ** 0.5)


def picard(max_sweeps: int):
    """Return (T, u, sweeps_done, residual history)."""
    gfu = ngs.GridFunction(V_u)          # zero -> k = K0 on the first sweep
    gfT = thermal_solve(gfu)
    gfu = structural_solve(gfT)
    hist = []
    for sweep in range(2, max_sweeps + 1):
        T_new = thermal_solve(gfu)
        u_new = structural_solve(T_new)
        dT = rel_l2(T_new, gfT, vector=False)
        du = rel_l2(u_new, gfu, vector=True)
        hist.append((sweep, dT, du))
        gfT, gfu = T_new, u_new
        if dT < TOL and du < TOL:
            break
    return gfT, gfu, (hist[-1][0] if hist else 1), hist


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    print(f"thermal_space_type={V_T.type} structural_space_type={V_u.type}")

    # --- WRONG variant: stop after the first pair of solves ---------------
    T_bad, u_bad, n_bad, _ = picard(N_PICARD_SWEEPS)
    print(f"one_way_sweeps={n_bad}")

    # --- RIGHT variant: iterate to convergence ---------------------------
    T_ok, u_ok, n_ok, hist = picard(MAX_SWEEPS)
    for sweep, dT, du in hist:
        print(f"picard_sweep={sweep} rel_dT={dT:.3e} rel_du={du:.3e}")
    err = rel_l2(T_ok, T_bad, vector=False)
    print(f"picard_sweeps_to_converge={n_ok} final_rel_dT={hist[-1][1]:.3e} "
          f"final_rel_du={hist[-1][2]:.3e}")
    print(f"one_way_temperature_rel_error={err:.4%}")
    print(f"one_way_error_gt_5pct={err > 0.05}")
    print(f"one_way_error_lt_20pct={err < 0.20}")
    print(f"picard_converged_below_1e-4={hist[-1][1] < TOL and hist[-1][2] < TOL}")
    print(f"picard_sweeps_in_3_to_10={3 <= n_ok <= 10}")
    print(f"picard_residual_monotone="
          f"{all(hist[i][1] < hist[i - 1][1] for i in range(1, len(hist)))}")

    if not err > 0.05:
        print(f"FAIL: the one-way result is only {err:.4%} from the converged "
              f"one, so the feedback through k(u) is not being exercised",
              file=sys.stderr)
        ok = False
    if not err < 0.20:
        print(f"FAIL: the one-way error {err:.4%} exceeds the documented "
              f"5-20% band", file=sys.stderr)
        ok = False
    if not (hist[-1][1] < TOL and hist[-1][2] < TOL):
        print("FAIL: the Picard loop did not reach the 1e-4 stopping criterion",
              file=sys.stderr)
        ok = False
    if not 3 <= n_ok <= 10:
        print(f"FAIL: Picard needed {n_ok} sweeps, outside the documented 3-10",
              file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
