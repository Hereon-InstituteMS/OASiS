"""Tier-2: displacement-only thin plates shear-lock; the HHJ mixed form does not.

Claim: ngsolve hdivdiv#5 — a displacement-only thin-plate H1 GridFunction at
h/thickness > 100 shows shear locking, the centre deflection being 10-1000x
smaller than the analytic simply-supported plate result; HHJ removes locking by
taking the bending moment as primary unknown.

Wrong variant: the standard un-reduced displacement-only Reissner-Mindlin
functional, H1 deflection + VectorH1 rotations, order 2, maxh=0.3, thickness
1e-3 (h/t = 300).

Observed on NGSolve 6.2.2604 (2026-08-03), E=1 nu=0.3 q=1, kappa=5/6,
reference w_max = 0.00406235 q L^4 / D:
  * t = 1e-3 (h/t = 300): w_c / w_ref = 5.50e-02, i.e. 18.2x too small.
  * t = 1e-1 (h/t = 3)  : w_c / w_ref = 1.05 -- the same code is fine, so the
    deficit is a discretisation artefact and not thick-plate physics.
  * HHJ (HDivDiv moments + H1 deflection) at t = 1e-3: within 1% of the
    reference, thickness-independent.
"""
from __future__ import annotations

import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: a thin plate resolved with h/t = 300 by a displacement-only form
T_THIN = 1.0e-3
T_MODERATE = 1.0e-1
MAXH = 0.3
ORDER = 2

E, NU, Q = 1.0, 0.3, 1.0
KAPPA = 5.0 / 6.0
G_SHEAR = E / (2.0 * (1.0 + NU))
NAVIER_COEFF = 0.004062352659370942
BND = "bottom|right|top|left"


def bending_rigidity(t: float) -> float:
    return E * t ** 3 / (12.0 * (1.0 - NU ** 2))


def reissner_mindlin_displacement_only(t: float) -> float:
    """H1 deflection + VectorH1 rotations, full (un-reduced) shear energy."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=MAXH))
    D = bending_rigidity(t)
    Vw = ngs.H1(mesh, order=ORDER, dirichlet=BND)
    Vb = ngs.VectorH1(mesh, order=ORDER)          # soft (simple) support
    X = Vw * Vb
    (w, b), (v, d) = X.TnT()

    def strain(u):
        return 0.5 * (ngs.Grad(u) + ngs.Grad(u).trans)

    a = ngs.BilinearForm(X, symmetric=True)
    a += D * ((1.0 - NU) * ngs.InnerProduct(strain(b), strain(d))
              + NU * ngs.div(b) * ngs.div(d)) * ngs.dx
    a += KAPPA * G_SHEAR * t * ngs.InnerProduct(ngs.grad(w) - b,
                                                ngs.grad(v) - d) * ngs.dx
    a.Assemble()
    f = ngs.LinearForm(X)
    f += Q * v * ngs.dx
    f.Assemble()
    gf = ngs.GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    return abs(gf.components[0](mesh(0.5, 0.5)))


def hhj_mixed(t: float) -> float:
    """Moment tensor in HDivDiv as primary unknown -- locking-free."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=MAXH))
    D = bending_rigidity(t)
    Vm = ngs.HDivDiv(mesh, order=ORDER - 1, dirichlet=BND)
    Vw = ngs.H1(mesh, order=ORDER, dirichlet=BND)
    X = Vm * Vw
    (sig, ww), (tau, vv) = X.TnT()
    n = ngs.specialcf.normal(2)

    def compliance(s):
        return (1.0 / (D * (1.0 - NU))) \
            * (s - NU / (1.0 + NU) * ngs.Trace(s) * ngs.Id(2))

    a = ngs.BilinearForm(X, symmetric=True)
    a += ngs.InnerProduct(compliance(sig), tau) * ngs.dx
    a += ngs.InnerProduct(tau, ww.Operator("hesse")) * ngs.dx
    a += ngs.InnerProduct(sig, vv.Operator("hesse")) * ngs.dx
    a += -(tau * n * n) * (ngs.Grad(ww) * n) * ngs.dx(element_boundary=True)
    a += -(sig * n * n) * (ngs.Grad(vv) * n) * ngs.dx(element_boundary=True)
    a.Assemble()
    f = ngs.LinearForm(X)
    f += Q * vv * ngs.dx
    f.Assemble()
    gf = ngs.GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    return abs(gf.components[1](mesh(0.5, 0.5)))


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")

    # --- WRONG variant: displacement-only at h/t = 300 --------------------
    ref_thin = NAVIER_COEFF * Q / bending_rigidity(T_THIN)
    wc_lock = reissner_mindlin_displacement_only(T_THIN)
    factor = ref_thin / wc_lock
    print(f"h_over_t={MAXH / T_THIN:.1f} disp_only_centre={wc_lock:.6e} "
          f"kirchhoff_reference={ref_thin:.6e} locking_factor={factor:.2f}")
    print(f"h_over_t_gt_100={MAXH / T_THIN > 100.0}")
    print(f"disp_only_locking_factor_in_10_1000={10.0 <= factor <= 1000.0}")
    if not 10.0 <= factor <= 1000.0:
        print(f"FAIL: displacement-only thin plate is {factor:.2f}x too small, "
              f"outside the documented 10-1000x locking band", file=sys.stderr)
        ok = False

    # --- control: same code, moderate thickness ---------------------------
    ref_mod = NAVIER_COEFF * Q / bending_rigidity(T_MODERATE)
    wc_mod = reissner_mindlin_displacement_only(T_MODERATE)
    f_mod = ref_mod / wc_mod
    print(f"h_over_t={MAXH / T_MODERATE:.1f} disp_only_centre={wc_mod:.6e} "
          f"factor={f_mod:.3f}")
    print(f"disp_only_unlocked_at_low_h_over_t={0.8 <= f_mod <= 1.25}")
    print(f"locking_grows_with_h_over_t={factor > 10.0 * f_mod}")
    if not 0.8 <= f_mod <= 1.25:
        print(f"FAIL: the same displacement-only code is already off by "
              f"{f_mod:.3f}x at h/t={MAXH / T_MODERATE:.0f}, so the thin-plate "
              f"deficit cannot be attributed to locking", file=sys.stderr)
        ok = False

    # --- RIGHT variant: HHJ mixed, same thickness -------------------------
    wc_hhj = hhj_mixed(T_THIN)
    rel = abs(wc_hhj - ref_thin) / ref_thin
    print(f"hhj_centre={wc_hhj:.6e} rel_dev_from_reference={rel:.3e}")
    print(f"hhj_locking_free_within_2pct={rel < 0.02}")
    print(f"hhj_beats_disp_only={abs(wc_hhj - ref_thin) < abs(wc_lock - ref_thin)}")
    if rel >= 0.02:
        print(f"FAIL: HHJ at t={T_THIN:g} is off by {rel:.3%}, so it does not "
              f"remove locking", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
