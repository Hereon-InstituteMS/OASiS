"""Tier-2: a CLAMPED HHJ plate needs a dw/dn=0 term; w=0 alone gives simple support.

Claim: ngsolve hdivdiv#1 — for a clamped plate you must add a boundary term for
dw/dn = 0 (Nitsche penalty or a Lagrange multiplier on the skeleton). With ONLY
w = 0 you get the simply-supported solution and the centre deflection is ~3-4x
larger than the clamped reference.

Wrong variant: keep w = 0 in H1 but ALSO put a homogeneous Dirichlet on M_nn in
the HDivDiv space. The boundary M_nn dofs ARE the Lagrange multiplier for
dw/dn = 0 (they multiply the leftover boundary part of the element_boundary
moment integral), so constraining them removes the only dw/dn enforcement the
formulation has.

Observed on NGSolve 6.2.2604 (2026-08-03), maxh=0.15 order=2, E=1 nu=0.3 t=1
q=1 (D=0.0915751):
  * multiplier removed: w_centre = 4.437e-02 = the SIMPLY-SUPPORTED Navier value,
    3.22x the clamped reference 0.00126 q L^4/D = 1.3759e-02, and the boundary
    normal-derivative energy fraction is ~1 (dw/dn not enforced at all).
  * multiplier retained: w_centre = 1.386e-02, within 1% of the clamped
    reference, and the boundary normal-derivative energy fraction collapses.
"""
from __future__ import annotations

import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: also constrain M_nn, which deletes the dw/dn multiplier
CLAMPED_KILLS_MULTIPLIER = True

E, NU, T_PLATE, Q = 1.0, 0.3, 1.0, 1.0
D = E * T_PLATE ** 3 / (12.0 * (1.0 - NU ** 2))
BND = "bottom|right|top|left"
W_CLAMPED = 0.00126 * Q / D              # Timoshenko clamped square plate
W_SIMPLE = 0.004062352659370942 * Q / D  # Navier simply-supported square plate


def hhj_plate(maxh: float, order: int, kill_multiplier: bool):
    """HHJ plate. kill_multiplier=True puts M_nn=0 on the whole boundary."""
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    Vm = (ngs.HDivDiv(mesh, order=order - 1, dirichlet=BND) if kill_multiplier
          else ngs.HDivDiv(mesh, order=order - 1))
    Vw = ngs.H1(mesh, order=order, dirichlet=BND)
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
    gw = gf.components[1]
    wc = abs(gw(mesh(0.5, 0.5)))
    # Grad() inside a boundary integral is the SURFACE gradient, whose normal
    # component is identically 0 -- BoundaryFromVolumeCF lifts the full
    # volume gradient onto the boundary so dw/dn is actually measurable.
    grad_w = ngs.BoundaryFromVolumeCF(ngs.grad(gw))
    dn2 = abs(ngs.Integrate((grad_w * n) ** 2 * ngs.ds, mesh))
    length = abs(ngs.Integrate(ngs.CoefficientFunction(1.0) * ngs.ds, mesh))
    dwdn_rms = (dn2 / length) ** 0.5
    return wc, dwdn_rms / wc          # dimensionless: |dw/dn| relative to w_c/L


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    _mm = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
    print(f"moment_space_type={ngs.HDivDiv(_mm, order=1).type} "
          f"deflection_space_type={ngs.H1(_mm, order=2).type}")

    print(f"clamped_reference={W_CLAMPED:.6e} simply_supported_reference="
          f"{W_SIMPLE:.6e}")

    # --- WRONG variant: dw/dn multiplier deleted --------------------------
    wc_bad, dn_bad = hhj_plate(0.15, 2, CLAMPED_KILLS_MULTIPLIER)
    ratio = wc_bad / W_CLAMPED
    print(f"no_dwdn_term_centre={wc_bad:.6e} over_clamped_ref={ratio:.4f} "
          f"bnd_normal_deriv_fraction={dn_bad:.4e}")
    print(f"no_dwdn_term_3x_to_4x_clamped={3.0 <= ratio <= 4.0}")
    print(f"no_dwdn_term_is_simply_supported="
          f"{abs(wc_bad - W_SIMPLE) / W_SIMPLE < 0.02}")
    print(f"no_dwdn_term_bnd_dwdn_unenforced={dn_bad > 0.1}")
    if not 3.0 <= ratio <= 4.0:
        print(f"FAIL: dropping the dw/dn term gave {ratio:.4f}x the clamped "
              f"reference, not the documented 3-4x", file=sys.stderr)
        ok = False
    if abs(wc_bad - W_SIMPLE) / W_SIMPLE >= 0.02:
        print("FAIL: the no-dw/dn solution is not the simply-supported one",
              file=sys.stderr)
        ok = False
    if dn_bad <= 0.1:
        print("FAIL: boundary dw/dn already small without the multiplier",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: multiplier retained ------------------------------
    wc_ok, dn_ok = hhj_plate(0.15, 2, False)
    rel = abs(wc_ok - W_CLAMPED) / W_CLAMPED
    print(f"with_dwdn_term_centre={wc_ok:.6e} rel_dev_from_clamped={rel:.3e} "
          f"bnd_normal_deriv_fraction={dn_ok:.4e}")
    print(f"with_dwdn_term_matches_clamped_within_3pct={rel < 0.03}")
    print(f"with_dwdn_term_reduces_bnd_dwdn={dn_ok < 0.25 * dn_bad}")
    if rel >= 0.03:
        print(f"FAIL: clamped HHJ missed the clamped reference by {rel:.3%}",
              file=sys.stderr)
        ok = False
    if dn_ok >= 0.25 * dn_bad:
        print("FAIL: the moment multiplier did not weakly enforce dw/dn=0",
              file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
