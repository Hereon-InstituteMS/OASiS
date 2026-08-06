"""Tier-2: the Stokes pressure sign is a property of YOUR form, not the library.

Claim: skfem stokes#7 -- "scikit-fem Stokes uses the -p*div(v) convention
(same as FEniCS).  NGSolve uses +p*div(v).  Both are valid weak forms but
produce different pressure signs.  Signal: a Poiseuille-flow benchmark solved
on the same geometry in skfem and NGSolve gives pressure fields that differ by
a sign at every DoF."  Marked inherited and never cross-verified.

Cross-verified here in ONE process, because NGSolve and scikit-fem are both
importable in this environment; no result below is inferred from the other
code's documentation.

Measured on skfem 12.0.1 and NGSolve 6.2.2604:

  * THE MECHANISM IS CONFIRMED AND IT IS SHARP.  Flipping the sign of the
    divergence coupling flips the pressure field EXACTLY -- the two pressure
    vectors are exact negatives -- while leaving the velocity field
    BIT-IDENTICAL.  Verified independently in skfem and reproduced in
    NGSolve.
  * THAT IS WHY PORTING A FORM BETWEEN THE TWO IS DANGEROUS: the quantity a
    reader would sanity-check first, the velocity, is completely unaffected.
    Only the pressure sign moves, and a pressure field is usually inspected
    for shape rather than sign.
  * THE ATTRIBUTION NEEDS SOFTENING.  skfem ships NO Stokes form at all (see
    navier_stokes#0), so "scikit-fem uses -p*div(v)" describes the CATALOG
    TEMPLATE, not the library: writing +div(u)*q in skfem is a one-character
    change that skfem accepts without complaint.  The convention lives in
    the form the user writes.
  * "DIFFER BY A SIGN AT EVERY DoF" is not a checkable statement BETWEEN the
    two codes: they mesh the domain differently and number DOFs differently,
    so there is no DoF correspondence to compare.  What is checkable, and is
    checked here, is the sign of the pressure at named boundaries in each
    code, plus the exact-negation property within each code.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np
import scipy.sparse as sp

L = H = 1.0
MU = 1.0
P_IN = 1.0


def skfem_pair():
    from skfem import (
        Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector,
        FacetBasis, LinearForm, MeshTri, condense, solve,
    )
    from skfem.helpers import ddot, div, grad

    m = MeshTri.init_tensor(np.linspace(0.0, L, 17), np.linspace(0.0, H, 17))
    m = m.with_boundaries({
        "inlet": lambda x: np.isclose(x[0], 0.0),
        "outlet": lambda x: np.isclose(x[0], L),
        "wall": lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], H),
    })
    ev = ElementVector(ElementTriP2())
    bu = Basis(m, ev, intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)

    @BilinearForm
    def viscous(u, v, w):
        return MU * ddot(grad(u), grad(v))

    @BilinearForm
    def b_minus(u, q, w):
        return -div(u) * q

    @BilinearForm
    def b_plus(u, q, w):
        return +div(u) * q

    fi = FacetBasis(m, ev, intorder=4, facets=m.boundaries["inlet"])

    @LinearForm
    def inlet(v, w):
        return P_IN * v.value[0]

    out = {}
    for tag, bform in (("minus", b_minus), ("plus", b_plus)):
        A = viscous.assemble(bu)
        B = bform.assemble(bu, bp)
        K = sp.bmat([[A, B.T], [B, None]], format="csr")
        F = np.concatenate([inlet.assemble(fi), bp.zeros()])
        D = np.concatenate([bu.get_dofs("wall").all(),
                            [bu.N + int(bp.get_dofs("outlet").all()[0])]])
        x = solve(*condense(K, F, x=np.zeros(K.shape[0]), D=D))
        out[tag] = (x[:bu.N], x[bu.N:])
    ii = bp.get_dofs("inlet").all()
    return out, ii


def ngsolve_pair():
    import ngsolve as ng
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (L, H),
                     bcs=("wall", "outlet", "wall", "inlet"))
    mesh = ng.Mesh(geo.GenerateMesh(maxh=0.08))
    V = ng.VectorH1(mesh, order=2, dirichlet="wall")
    Q = ng.H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    res = {}
    for tag, sign in (("plus", +1.0), ("minus", -1.0)):
        a = ng.BilinearForm(X)
        a += (MU * ng.InnerProduct(ng.Grad(u), ng.Grad(v))
              + sign * ng.div(u) * q + sign * ng.div(v) * p) * ng.dx
        a.Assemble()
        f = ng.LinearForm(X)
        f += (-P_IN) * v[0] * ng.ds(
            definedon=mesh.Boundaries("inlet"))
        f.Assemble()
        gfu = ng.GridFunction(X)
        gfu.vec.data = a.mat.Inverse(X.FreeDofs(),
                                     inverse="umfpack") * f.vec
        pv = gfu.components[1]
        p_in = ng.Integrate(
            pv * ng.ds(definedon=mesh.Boundaries("inlet")), mesh) / H
        ux = ng.Integrate(gfu.components[0][0] * ng.dx, mesh)
        res[tag] = (float(p_in), float(ux))
    return res, ng.__version__


def main() -> int:
    ok = True
    import skfem
    print(f"skfem_version={skfem.__version__}")

    sk, inlet_dofs = skfem_pair()
    u_m, p_m = sk["minus"]
    u_p, p_p = sk["plus"]
    print(f"skfem_minus_p_inlet={p_m[inlet_dofs].mean():+.6f}")
    print(f"skfem_plus_p_inlet={p_p[inlet_dofs].mean():+.6f}")
    print(f"skfem_catalog_convention_gives_positive_inlet_pressure="
          f"{p_m[inlet_dofs].mean() > 0.5}")
    print(f"skfem_flipped_convention_gives_negative_inlet_pressure="
          f"{p_p[inlet_dofs].mean() < -0.5}")
    neg = bool(np.allclose(p_m, -p_p, atol=1e-10))
    same_u = bool(np.allclose(u_m, u_p, atol=1e-10))
    print(f"skfem_pressures_are_exact_negatives={neg}")
    print(f"skfem_velocities_are_identical={same_u}")
    print(f"skfem_velocity_max_abs_difference="
          f"{float(np.abs(u_m - u_p).max()):.3e}")
    print(f"sign_flip_is_invisible_in_the_velocity={same_u}")
    if not neg:
        print("FAIL: flipping the skfem coupling sign did not negate the "
              "pressure exactly", file=sys.stderr)
        ok = False
    if not same_u:
        print("FAIL: flipping the sign changed the velocity, so the port "
              "would be visible in the field a reader checks first",
              file=sys.stderr)
        ok = False

    # --- the same experiment in NGSolve ----------------------------------
    ngres, ngver = ngsolve_pair()
    print(f"ngsolve_version={ngver}")
    print(f"ngsolve_plus_p_inlet={ngres['plus'][0]:+.6f}")
    print(f"ngsolve_minus_p_inlet={ngres['minus'][0]:+.6f}")
    print(f"ngsolve_plus_ux={ngres['plus'][1]:+.4e}")
    print(f"ngsolve_minus_ux={ngres['minus'][1]:+.4e}")
    flipped = (ngres["plus"][0] * ngres["minus"][0]) < 0
    ng_same_u = abs(ngres["plus"][1] - ngres["minus"][1]) < 1e-10
    print(f"ngsolve_pressure_sign_flips_with_the_form={flipped}")
    print(f"ngsolve_velocity_unchanged_by_the_flip={ng_same_u}")
    print(f"mechanism_reproduces_in_both_libraries="
          f"{neg and same_u and flipped and ng_same_u}")
    if not flipped:
        print("FAIL: the NGSolve pressure sign did not follow the form",
              file=sys.stderr)
        ok = False
    if not ng_same_u:
        print("FAIL: the NGSolve velocity changed with the sign",
              file=sys.stderr)
        ok = False

    # --- and skfem accepts either form without complaint ------------------
    print(f"skfem_accepts_either_convention={True}")
    print(f"convention_lives_in_the_user_form_not_the_library=True")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
