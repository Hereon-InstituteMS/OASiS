"""Tier-2: the SS square-plate reference is the Navier series, and the shipped
hdivdiv_2d template is ~78% below it for two separable reasons.

Claim: ngsolve hdivdiv#6 — w_max = 0.00406235 q L^4 / D for a simply-supported
square plate; q L^4/(64 D) = 0.015625 q L^4/D is the CLAMPED CIRCULAR plate and
is 3.85x too large. Open issue in the same entry: even against the corrected
reference the shipped hdivdiv_2d template is still ~78% low.

Wrong variant: this fixture runs the shipped template's own bilinear form,
verbatim in the two respects that matter:
  * Compliance(s) = (1/D)(s - nu/(1+nu) Tr(s) I) -- the factor 1/(1-nu) of the
    inverse plate stiffness is missing;
  * HDivDiv(mesh, order=order-1) with NO dirichlet -- so M_nn is unconstrained
    on the boundary and its free dofs weakly enforce dw/dn = 0, i.e. the solve
    silently returns a CLAMPED plate.

Observed on NGSolve 6.2.2604 (2026-08-03), maxh=0.15 order=2, E=1 nu=0.3 t=1
q=1 (D=0.0915751, Navier reference 4.436089e-02):
    shipped (both defects)      w_c/w_ref = 0.2187   -> 78.1% low
    + M_nn Dirichlet only       w_c/w_ref = 0.7002   = 1 - nu exactly
    + compliance factor only    w_c/w_ref = 0.3124   = clamped/simply-supported
    both fixes                  w_c/w_ref = 1.0002
0.7002 * 0.3124 = 0.2188, so the two defects are independent and multiplicative.
"""
from __future__ import annotations

import math
import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: the two defects of the shipped hdivdiv_2d template
SHIPPED_COMPLIANCE_MISSES_1_MINUS_NU = True
SHIPPED_OMITS_MNN_DIRICHLET = True

E, NU, T_PLATE, Q = 1.0, 0.3, 1.0, 1.0
D = E * T_PLATE ** 3 / (12.0 * (1.0 - NU ** 2))
BND = "bottom|right|top|left"


def navier_coefficient(n_terms: int = 101) -> float:
    """w_max * D / (q L^4) for a simply-supported square plate."""
    return (16.0 / math.pi ** 6) * sum(
        math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
        / (m * n * (m ** 2 + n ** 2) ** 2)
        for m in range(1, n_terms, 2) for n in range(1, n_terms, 2))


def hhj_plate(maxh: float, order: int, drop_1_minus_nu: bool,
              omit_mnn_dirichlet: bool) -> tuple[float, str, str]:
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    Vm = (ngs.HDivDiv(mesh, order=order - 1) if omit_mnn_dirichlet
          else ngs.HDivDiv(mesh, order=order - 1, dirichlet=BND))
    Vw = ngs.H1(mesh, order=order, dirichlet=BND)
    X = Vm * Vw
    (sig, ww), (tau, vv) = X.TnT()
    n = ngs.specialcf.normal(2)
    stiff = D if drop_1_minus_nu else D * (1.0 - NU)

    def compliance(s):
        return (1.0 / stiff) * (s - NU / (1.0 + NU) * ngs.Trace(s) * ngs.Id(2))

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
    return abs(gf.components[1](mesh(0.5, 0.5))), Vm.type, Vw.type


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")

    coeff = navier_coefficient()
    w_ref = coeff * Q / D
    old = Q / (64.0 * D)
    print(f"navier_series_coefficient={coeff:.8f} navier_w_max={w_ref:.8f}")
    print(f"old_formula_qL4_over_64D={old:.8f} "
          f"old_over_navier={old / w_ref:.4f}")
    print(f"navier_matches_textbook_0p00406={abs(coeff - 0.00406) < 5e-6}")
    print(f"old_formula_over_navier_gt_3p8={old / w_ref > 3.8}")
    print(f"old_formula_coefficient_is_0p015625="
          f"{abs(old * D / Q - 0.015625) < 1e-9}")
    if abs(coeff - 0.00406) >= 5e-6:
        print(f"FAIL: Navier series gives {coeff:.8f}, not the textbook 0.00406",
              file=sys.stderr)
        ok = False
    if old / w_ref <= 3.8:
        print("FAIL: q L^4/(64 D) is not >3.8x the Navier value", file=sys.stderr)
        ok = False

    # --- WRONG variant: the shipped template's own form -------------------
    wc_ship, m_type, w_type = hhj_plate(
        0.15, 2, SHIPPED_COMPLIANCE_MISSES_1_MINUS_NU, SHIPPED_OMITS_MNN_DIRICHLET)
    r_ship = wc_ship / w_ref
    print(f"moment_space_type={m_type} deflection_space_type={w_type}")
    print(f"shipped_template_centre={wc_ship:.8f} over_navier={r_ship:.4f} "
          f"low_by={1.0 - r_ship:.1%}")
    print(f"shipped_template_low_by_gt_70pct={1.0 - r_ship > 0.70}")
    print(f"shipped_template_low_by_lt_85pct={1.0 - r_ship < 0.85}")
    print(f"shipped_template_within_5pct_of_navier={abs(r_ship - 1.0) < 0.05}")
    if not 0.70 < 1.0 - r_ship < 0.85:
        print(f"FAIL: the shipped hdivdiv_2d form is {1.0 - r_ship:.1%} low, "
              f"not the recorded ~78%", file=sys.stderr)
        ok = False

    # --- attribute the deficit to two independent defects -----------------
    wc_mnn = hhj_plate(0.15, 2, True, False)[0] / w_ref
    wc_cmp = hhj_plate(0.15, 2, False, True)[0] / w_ref
    print(f"only_mnn_dirichlet_added_ratio={wc_mnn:.4f} (expect 1-nu="
          f"{1.0 - NU:.4f})")
    print(f"only_compliance_fixed_ratio={wc_cmp:.4f}")
    print(f"mnn_fix_alone_gives_1_minus_nu={abs(wc_mnn - (1.0 - NU)) < 0.01}")
    print(f"defects_are_multiplicative="
          f"{abs(wc_mnn * wc_cmp - r_ship) < 0.01}")
    if abs(wc_mnn - (1.0 - NU)) >= 0.01:
        print(f"FAIL: fixing only the M_nn BC gave {wc_mnn:.4f}, not 1-nu",
              file=sys.stderr)
        ok = False
    if abs(wc_mnn * wc_cmp - r_ship) >= 0.01:
        print("FAIL: the two defects are not independent/multiplicative",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: both defects repaired -----------------------------
    wc_fix = hhj_plate(0.15, 2, False, False)[0]
    rel = abs(wc_fix - w_ref) / w_ref
    print(f"corrected_centre={wc_fix:.8f} rel_dev_from_navier={rel:.3e}")
    print(f"corrected_matches_navier_within_1pct={rel < 0.01}")
    if rel >= 0.01:
        print(f"FAIL: the corrected HHJ form is off the Navier reference by "
              f"{rel:.3%}", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
