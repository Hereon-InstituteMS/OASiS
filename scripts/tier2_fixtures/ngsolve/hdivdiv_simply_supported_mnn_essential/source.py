"""Tier-2: for a SIMPLY-SUPPORTED HHJ plate M_nn=0 is ESSENTIAL, not natural.

Claim: ngsolve hdivdiv#2 — "only w = 0 on the boundary is needed; normal moments
M_nn vanish naturally as a natural BC. Adding an extra Dirichlet on M_nn
over-constrains a simply-supported plate, increasing the stiffness — centre
deflection is ~10-20% smaller than the analytic Navier value 0.00406 q L^4 / D".

Both halves of that signal are FALSE on NGSolve 6.2.2604 and this fixture pins
what actually happens (measured 2026-08-03, maxh=0.15 order=2, E=1 nu=0.3 t=1
q=1, D=0.0915751, Navier reference 4.436089e-02):

  * WITH the M_nn Dirichlet on HDivDiv:   w_c = 4.4371e-02 -> 1.0002x Navier.
    So the "extra" Dirichlet is not extra: it IS the simply-supported condition,
    and it makes the plate SOFTER, not stiffer.
  * WITHOUT it ("M_nn vanishes naturally"): w_c = 1.3860e-02 -> 0.3124x Navier,
    i.e. 69% too SMALL, because the free boundary M_nn dofs act as the Lagrange
    multiplier for dw/dn = 0 and silently clamp the plate. The measured boundary
    M_nn is O(1) there and ~1e-15 with the Dirichlet, so the BC is essential.
  * The predicted "~10-20% smaller" band (ratio 0.80-0.90) is not reached by
    either variant.

Wrong variant this fixture executes: the omitted M_nn Dirichlet, i.e. the
"only w = 0 is needed" recipe.
"""
from __future__ import annotations

import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: trust "M_nn vanishes naturally" and leave the HDivDiv space unconstrained
SS_SETS_MNN_DIRICHLET = False

E, NU, T_PLATE, Q = 1.0, 0.3, 1.0, 1.0
D = E * T_PLATE ** 3 / (12.0 * (1.0 - NU ** 2))
BND = "bottom|right|top|left"
W_NAVIER = 0.004062352659370942 * Q / D    # simply-supported square plate
W_OLD_CATALOG = Q / (64.0 * D)             # clamped CIRCULAR plate -- wrong here


def hhj_plate(maxh: float, order: int, mnn_dirichlet: bool):
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    Vm = (ngs.HDivDiv(mesh, order=order - 1, dirichlet=BND) if mnn_dirichlet
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
    gsig, gw = gf.components
    wc = abs(gw(mesh(0.5, 0.5)))
    # boundary M_nn, lifted from the volume (Trace() of an HDivDiv GridFunction
    # in a ds integral is the surface object, so lift explicitly)
    sig_b = ngs.BoundaryFromVolumeCF(gsig)
    mnn2 = abs(ngs.Integrate(ngs.InnerProduct(sig_b * n, n) ** 2 * ngs.ds, mesh))
    scale = abs(ngs.Integrate(ngs.InnerProduct(gsig, gsig) * ngs.dx, mesh))
    return wc, (mnn2 / scale) ** 0.5, int(sum(Vm.FreeDofs()))


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    print(f"navier_ss_reference={W_NAVIER:.6e} "
          f"old_catalog_qL4_over_64D={W_OLD_CATALOG:.6e} "
          f"old_over_navier={W_OLD_CATALOG / W_NAVIER:.4f}")

    # --- WRONG variant: rely on "M_nn vanishes naturally" -----------------
    wc_bad, mnn_bad, free_bad = hhj_plate(0.15, 2, SS_SETS_MNN_DIRICHLET)
    r_bad = wc_bad / W_NAVIER
    print(f"no_mnn_dirichlet_centre={wc_bad:.6e} over_navier={r_bad:.4f} "
          f"rel_bnd_mnn={mnn_bad:.3e} moment_free_dofs={free_bad}")
    print(f"no_mnn_dirichlet_more_than_2x_too_small={r_bad < 0.5}")
    print(f"no_mnn_dirichlet_leaves_bnd_mnn_nonzero={mnn_bad > 1e-3}")
    # the band the catalog predicted for the OPPOSITE variant
    print(f"catalog_10_to_20pct_smaller_band_reached={0.80 <= r_bad <= 0.90}")
    if r_bad >= 0.5:
        print(f"FAIL: omitting the M_nn Dirichlet did not stiffen the plate "
              f"(ratio {r_bad:.4f}); M_nn may now really be natural",
              file=sys.stderr)
        ok = False
    if mnn_bad <= 1e-3:
        print("FAIL: boundary M_nn already vanishes without the Dirichlet",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: M_nn = 0 imposed essentially ----------------------
    wc_ok, mnn_ok, free_ok = hhj_plate(0.15, 2, True)
    rel = abs(wc_ok - W_NAVIER) / W_NAVIER
    print(f"with_mnn_dirichlet_centre={wc_ok:.6e} rel_dev_from_navier={rel:.3e} "
          f"rel_bnd_mnn={mnn_ok:.3e} moment_free_dofs={free_ok}")
    print(f"with_mnn_dirichlet_matches_navier_within_1pct={rel < 0.01}")
    print(f"with_mnn_dirichlet_zeroes_bnd_mnn={mnn_ok < 1e-10}")
    print(f"mnn_dirichlet_makes_plate_softer={wc_ok > wc_bad}")
    print(f"mnn_dirichlet_removes_free_dofs={free_ok < free_bad}")
    if rel >= 0.01:
        print(f"FAIL: simply-supported HHJ missed the Navier reference by "
              f"{rel:.3%}", file=sys.stderr)
        ok = False
    if mnn_ok >= 1e-10:
        print("FAIL: the M_nn Dirichlet did not act as an essential BC",
              file=sys.stderr)
        ok = False
    if not wc_ok > wc_bad:
        print("FAIL: the M_nn Dirichlet did not soften the plate", file=sys.stderr)
        ok = False
    if not free_ok < free_bad:
        print("FAIL: the M_nn Dirichlet removed no dofs from HDivDiv",
              file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
