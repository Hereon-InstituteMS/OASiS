"""Tier-2: the phase-field peak load is set by l0, not by h - refining the
mesh at fixed l0 changes nothing at all.

Claim: ngsolve phase_field#6 - l0 must be small enough relative to the
specimen; refining the mesh WITHOUT also reducing l0 does not help.

Wrong variant: chase a converged peak load by halving maxh while leaving
  l0 = 0.1 alone.
Right variant : halve l0.  For the AT2 model the homogeneous response has
  the closed form
      d = 2 psi / (Gc/l0 + 2 psi),   sigma = (1-d)^2 E* eps
  whose maximum over eps is  sigma_c = sqrt(27 E* Gc / (256 l0))  at
  eps_c = sqrt(Gc / (3 E* l0)),  so sigma_c ~ l0^(-1/2) and h does not enter.

Confined uniaxial tension is used (dirichletx='left|right',
dirichlety='bottom|top') so the strain state is uniform and the numerical
peak can be compared with the closed form directly.

Observed on NGSolve 6.2.2604, unit_square, VectorH1/H1 order 1, E=1, nu=0.3
(so E* = lam + 2 mu = 1.34615385), Gc=1e-3, load scanned in 80 steps up to
eps=0.16 (2026-08-06):
  maxh=0.2, l0=0.10 (ne=52,  ndof_u=74)  peak sigma 0.037679, analytic
                                          0.037680, at eps 0.0500
  maxh=0.1, l0=0.10 (ne=230, ndof_u=272) peak sigma 0.037679 - the SAME
                                          value to every printed digit
  maxh=0.1, l0=0.05                       peak sigma 0.053286, analytic
                                          0.053287, ratio to l0=0.1 is
                                          1.4142 = sqrt(2)
"""
from __future__ import annotations

import math
import sys

from netgen.geom2d import unit_square
from ngsolve import (BilinearForm, CoefficientFunction, GridFunction, Grad,
                     H1, Id, Integrate, InnerProduct, LinearForm, Mesh, Trace,
                     VectorH1, dx, grad)

E_MOD, NU = 1.0, 0.3
MU = E_MOD / (2 * (1 + NU))
LAM = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
ESTAR = LAM + 2 * MU
GC, KRES = 1e-3, 1e-10
NSTEPS, EMAX = 80, 0.16
L0_BIG, L0_SMALL = 0.10, 0.05


def strain(w):
    return 0.5 * (Grad(w) + Grad(w).trans)


def analytic_peak(l0):
    return math.sqrt(27 * ESTAR * GC / (256 * l0))


def peak_stress(maxh, l0):
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    Vu = VectorH1(mesh, order=1, dirichletx="left|right",
                  dirichlety="bottom|top")
    Vd = H1(mesh, order=1)
    u, v = Vu.TnT()
    d, phi = Vd.TnT()
    area = Integrate(CoefficientFunction(1) * dx, mesh)
    best_sig, best_eps = 0.0, 0.0
    for k in range(1, NSTEPS + 1):
        disp = EMAX * k / NSTEPS
        gfu, gfd = GridFunction(Vu), GridFunction(Vd)
        gfd.Set(0)
        for _ in range(30):
            gfu.Set(CoefficientFunction((0, disp)),
                    definedon=mesh.Boundaries("top"))
            a = BilinearForm(Vu)
            a += ((1 - gfd) ** 2 + KRES) * InnerProduct(
                2 * MU * strain(u) + LAM * Trace(strain(u)) * Id(2),
                strain(v)) * dx
            a.Assemble()
            f = LinearForm(Vu)
            f.Assemble()
            r = f.vec.CreateVector()
            r.data = f.vec - a.mat * gfu.vec
            gfu.vec.data += a.mat.Inverse(Vu.FreeDofs()) * r
            e = strain(gfu)
            hfield = 0.5 * LAM * Trace(e) ** 2 + MU * InnerProduct(e, e)
            ad = BilinearForm(Vd, symmetric=True)
            ad += ((GC / l0 + 2 * hfield) * d * phi
                   + GC * l0 * grad(d) * grad(phi)) * dx
            ad.Assemble()
            fd = LinearForm(2 * hfield * phi * dx)
            fd.Assemble()
            gnew = GridFunction(Vd)
            gnew.vec.data = ad.mat.Inverse(Vd.FreeDofs()) * fd.vec
            delta = (gnew.vec - gfd.vec).Norm()
            gfd.vec.data = gnew.vec
            if delta < 1e-11:
                break
        e = strain(gfu)
        sig = ((1 - gfd) ** 2 + KRES) * (2 * MU * e
                                          + LAM * Trace(e) * Id(2))
        s = Integrate(sig[1, 1] * dx, mesh) / area
        if s > best_sig:
            best_sig, best_eps = s, disp
    return best_sig, best_eps, mesh.ne, Vu.ndof


def main() -> int:
    ok = True
    print(f"vector_space_class=VectorH1 estar_is_lam_plus_2mu="
          f"{abs(ESTAR - (LAM + 2 * MU)) < 1e-15}")

    sig_a, eps_a, ne_a, nd_a = peak_stress(0.2, L0_BIG)
    sig_b, eps_b, ne_b, nd_b = peak_stress(0.1, L0_BIG)
    sig_c, eps_c, ne_c, nd_c = peak_stress(0.1, L0_SMALL)
    print(f"coarse_mesh_ne={ne_a} ndof_u={nd_a}")
    print(f"fine_mesh_ne={ne_b} ndof_u={nd_b}")
    print(f"peak_l0_0p10_maxh0p2={sig_a:.6f} at_eps={eps_a:.4f}")
    print(f"peak_l0_0p10_maxh0p1={sig_b:.6f} at_eps={eps_b:.4f}")
    print(f"peak_l0_0p05_maxh0p1={sig_c:.6f} at_eps={eps_c:.4f}")
    print(f"mesh_was_actually_refined={ne_b > 3 * ne_a}")

    for tag, sig, l0 in (("l0_0p10_maxh0p2", sig_a, L0_BIG),
                         ("l0_0p10_maxh0p1", sig_b, L0_BIG),
                         ("l0_0p05_maxh0p1", sig_c, L0_SMALL)):
        rel = abs(sig / analytic_peak(l0) - 1)
        print(f"peak_matches_analytic_{tag}={rel < 0.01}")
        if rel >= 0.01:
            print(f"FAIL: {tag} peak {sig:.6f} is {rel * 100:.2f}% off the "
                  f"closed form {analytic_peak(l0):.6f}", file=sys.stderr)
            ok = False

    h_effect = abs(sig_b / sig_a - 1)
    l0_effect = abs(sig_c / sig_b - 1)
    print(f"refining_h_at_fixed_l0_changes_peak_lt_1pct={h_effect < 0.01}")
    print(f"halving_l0_changes_peak_gt_20pct={l0_effect > 0.20}")
    print("peak_ratio_matches_inverse_sqrt_l0_within_2pct="
          f"{abs((sig_c / sig_b) / math.sqrt(2.0) - 1) < 0.02}")
    print(f"l0_effect_dominates_h_effect={l0_effect > 20 * h_effect}")
    if not h_effect < 0.01:
        print(f"FAIL: refining the mesh at fixed l0 moved the peak by "
              f"{h_effect * 100:.2f}%", file=sys.stderr)
        ok = False
    if not l0_effect > 0.20:
        print(f"FAIL: halving l0 moved the peak by only "
              f"{l0_effect * 100:.2f}%", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
