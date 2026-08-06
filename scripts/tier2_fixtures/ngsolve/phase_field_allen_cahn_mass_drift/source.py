"""Tier-2: Allen-Cahn does not conserve mass; the mixed H1xH1 Cahn-Hilliard
form conserves it to machine precision.

Claim: ngsolve phase_field#0 - Integrate(c, mesh) drifts monotonically in
Allen-Cahn (~1-5% per characteristic interface time); the same geometry
under Cahn-Hilliard preserves the integral to machine precision.

Wrong variant: the shipped phase_field_2d scheme - semi-implicit Allen-Cahn,
  dc/dt = M(eps^2 Lap c - W'(c)), W'(c) = 2c(1-c)(1-2c), on a shrinking
  circular droplet.  Nothing in the formulation constrains Integrate(c).
Right variant : mixed (c, mu) on H1 x H1 with the same mesh, eps, M and dt.
  Testing the c-equation with q = 1 gives Integrate(c - c_old) = 0
  identically, so the drift is round-off only.

Observed on NGSolve 6.2.2604, unit_square maxh=0.06 (ne=634, P1 ndof=352),
eps=0.06, droplet radius 0.3, M=20, dt=2e-4, 60 steps (2026-08-06):
  Allen-Cahn     relative mass drift = -2.755e-02, monotone decreasing
  Cahn-Hilliard  relative mass drift = -3.516e-16
Both fields stay in [0, 1] to within a few percent, so the drift is genuine
curvature-driven shrinkage and not a blow-up.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import (BilinearForm, CoefficientFunction, GridFunction, H1,
                     Integrate, LinearForm, Mesh, dx, exp, grad, sqrt, x, y)

EPS, MOB, DT, NSTEPS, RAD, STAB = 0.06, 20.0, 2e-4, 60, 0.30, 2.0


def w_prime(c):
    return 2 * c * (1 - c) * (1 - 2 * c)


def droplet():
    r = sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    e2 = exp(-2 * (r - RAD) / (2 * EPS))
    return 0.5 + 0.5 * (e2 - 1) / (e2 + 1)


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.06))
    fes = H1(mesh, order=1)
    c, w = fes.TnT()
    c0 = droplet()
    area = Integrate(CoefficientFunction(1) * dx, mesh)
    print(f"mesh_ne={mesh.ne} ac_ndof={fes.ndof}")

    # ---- WRONG: semi-implicit Allen-Cahn (the shipped scheme) -----------
    lhs = BilinearForm(((1 / DT) * c * w
                        + EPS ** 2 * MOB * grad(c) * grad(w)) * dx).Assemble()
    inv = lhs.mat.Inverse(fes.FreeDofs())
    gfc = GridFunction(fes)
    gfc.Set(c0)
    m_ac = [Integrate(gfc, mesh)]
    for _ in range(NSTEPS):
        rhs = LinearForm(((1 / DT) * gfc - MOB * w_prime(gfc)) * w * dx)
        rhs.Assemble()
        gfc.vec.data = inv * rhs.vec
        m_ac.append(Integrate(gfc, mesh))
    ac_drift = (m_ac[-1] - m_ac[0]) / m_ac[0]
    ac_mono = all(m_ac[i + 1] < m_ac[i] for i in range(len(m_ac) - 1))
    ac_lo, ac_hi = min(gfc.vec), max(gfc.vec)
    print(f"ac_mass_drift_gt_1pct={abs(ac_drift) > 0.01}")
    print(f"ac_mass_monotone_decreasing={ac_mono}")
    print(f"ac_field_stays_bounded={-0.05 < ac_lo and ac_hi < 1.05}")
    if not (abs(ac_drift) > 0.01 and ac_mono):
        print(f"FAIL: Allen-Cahn mass drift was {ac_drift:.3e}, monotone="
              f"{ac_mono}; the non-conservation pitfall is absent",
              file=sys.stderr)
        ok = False
    if not (-0.05 < ac_lo and ac_hi < 1.05):
        print(f"FAIL: the Allen-Cahn field left [0,1] "
              f"({ac_lo:.4f}, {ac_hi:.4f}); the drift is a blow-up, not "
              f"curvature-driven shrinkage", file=sys.stderr)
        ok = False

    # ---- RIGHT: mixed (c, mu) Cahn-Hilliard -----------------------------
    X = fes * fes
    (cc, mu), (q, p) = X.TnT()
    a = BilinearForm(X)
    a += ((1 / DT) * cc * q + MOB * grad(mu) * grad(q)
          + mu * p - STAB * cc * p) * dx
    a += -EPS ** 2 * grad(cc) * grad(p) * dx
    a.Assemble()
    invX = a.mat.Inverse(X.FreeDofs())
    gf = GridFunction(X)
    gf.components[0].Set(c0)
    m_ch = [Integrate(gf.components[0], mesh)]
    for _ in range(NSTEPS):
        cold = gf.components[0]
        rhs = LinearForm(((1 / DT) * cold * q
                          + (w_prime(cold) - STAB * cold) * p) * dx)
        rhs.Assemble()
        gf.vec.data = invX * rhs.vec
        m_ch.append(Integrate(gf.components[0], mesh))
    ch_drift = (m_ch[-1] - m_ch[0]) / m_ch[0]
    ch_lo, ch_hi = min(gf.components[0].vec), max(gf.components[0].vec)
    print(f"ch_space_class={type(X).__name__}")
    print(f"ch_space_ncomponents={len(X.components)}")
    print(f"ch_ndof_is_twice_ac_ndof={X.ndof == 2 * fes.ndof}")
    print(f"ch_mass_drift_lt_1em12={abs(ch_drift) < 1e-12}")
    print(f"ch_field_stays_bounded={-0.05 < ch_lo and ch_hi < 1.05}")
    print("ch_conserves_better_than_ac_by_gt_1e10="
          f"{abs(ac_drift) / max(abs(ch_drift), 1e-300) > 1e10}")
    if not abs(ch_drift) < 1e-12:
        print(f"FAIL: Cahn-Hilliard drifted by {ch_drift:.3e}, not machine "
              f"precision", file=sys.stderr)
        ok = False
    if not (-0.05 < ch_lo and ch_hi < 1.05):
        print(f"FAIL: the Cahn-Hilliard field left [0,1] "
              f"({ch_lo:.4f}, {ch_hi:.4f}); mass conservation of a blown-up "
              f"solution proves nothing", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
