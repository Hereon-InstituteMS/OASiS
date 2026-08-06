"""Tier-2: forgetting the ElementDG wrapper silently turns a DG form into CG.

Claim: skfem dg_methods#0 -- ElementDG wraps any element to make it fully
discontinuous.  Forgetting the wrapper and using a bare ElementTriP1 in a DG
context produces a continuous-Galerkin global solution -- no jumps at element
edges, and the upwind/penalty terms in the form evaluate to zero.

Wrong variant: assemble the identical upwind-flux and interior-penalty forms
over InteriorFacetBasis built on a BARE ElementTriP1, then solve the same
steady advection problem with it.

Observed on skfem 12.0.1 (MeshTri().refined(3), 128 elements, 81 vertices):
  * bare ElementTriP1 gives Basis.N = 81 = one DOF per vertex (continuous);
    ElementDG(ElementTriP1()) gives Basis.N = 384 = 3 x 128 (element-local).
  * the upwind interior-facet matrix assembled on the bare element has
    max|entry| = 5.6e-17, i.e. it is the zero matrix; on the wrapped element
    it is 6.2e-02.  Same for the interior-penalty matrix (1.7e-16 vs 2.0e-01).
  * the bare-element solution has max|u+ - u-| = 0 across every interior
    facet, while the DG solution jumps by O(0.1) at the advected front.

Mutation control: T2_MUTATE=1 hands the wrong-variant probe
ElementDG(ElementTriP1()) instead of the bare ElementTriP1() -- the documented
fix, applied at the element site. The "bare" leg then has 384 DOFs and live
facet matrices, so bare_p1_n_dofs=81, bare_p1_equals_n_vertices=True,
bare_p1_upwind_flux_matrix_is_zero=True, bare_p1_penalty_matrix_is_zero=True and
bare_p1_solution_has_no_jumps=True all vanish and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementDG,
    ElementTriP1,
    FacetBasis,
    InteriorFacetBasis,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import jump
from scipy.sparse.linalg import spsolve

MUTATE = os.environ.get("T2_MUTATE") == "1"

B = np.array([1.0, 0.5])


@BilinearForm
def advection_volume(u, v, w):
    return (B[0] * u.grad[0] + B[1] * u.grad[1]) * v


@BilinearForm
def upwind_flux(u, v, w):
    """Single-sided upwind flux over the side=0/side=1 InteriorFacetBasis pair."""
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    sv = (-1.0) ** w.idx[1]
    ju = (-1.0) ** w.idx[0] * u
    return np.minimum(sv * bn, 0.0) * (-sv) * ju * v


@BilinearForm
def interior_penalty(u, v, w):
    ju, jv = jump(w, u, v)
    return ju * jv / w.h


@BilinearForm
def inflow_bilinear(u, v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    return -np.minimum(bn, 0.0) * u * v


def inflow_linear(gfun):
    @LinearForm
    def form(v, w):
        bn = B[0] * w.n[0] + B[1] * w.n[1]
        return -np.minimum(bn, 0.0) * gfun(w.x) * v
    return form


def g_step(x):
    return np.where(x[1] < 0.5, 1.0, 0.0)


def probe(m, element):
    """Return DOF count, |flux matrix|, |penalty matrix| and the solution jump."""
    ib = Basis(m, element)
    i0 = InteriorFacetBasis(m, element, side=0)
    i1 = InteriorFacetBasis(m, element, side=1)
    fb = FacetBasis(m, element)

    flux = asm(upwind_flux, [i0, i1], [i0, i1])
    pen = asm(interior_penalty, [i0, i1], [i0, i1])

    A = asm(advection_volume, ib) + flux + asm(inflow_bilinear, fb)
    f = asm(inflow_linear(g_step), fb)
    u = spsolve(A.tocsr(), f)

    # interpolate() returns a DiscreteField (an ndarray subclass) of shape
    # (n_interior_facets, n_quadrature_points); side=0 and side=1 walk the same
    # facet list in the same order, so the difference IS the facet jump.
    v0 = np.asarray(i0.interpolate(u))
    v1 = np.asarray(i1.interpolate(u))
    return {
        "n_dofs": int(ib.N),
        "flux_max": float(abs(flux).max()),
        "pen_max": float(abs(pen).max()),
        "sol_jump_max": float(np.abs(v0 - v1).max()),
        "u": u,
    }


def main() -> int:
    ok = True
    m = MeshTri().refined(3)
    print(f"n_elements={m.nelements}")
    print(f"n_vertices={m.p.shape[1]}")

    # --- WRONG variant: no ElementDG wrapper ------------------------------
    # Pathology: the DG forms are built on a bare (continuous) ElementTriP1.
    # Mutation: the documented fix -- wrap the element in ElementDG.
    bare = probe(m, ElementDG(ElementTriP1()) if MUTATE else ElementTriP1())
    print(f"bare_p1_n_dofs={bare['n_dofs']}")
    print(f"bare_p1_equals_n_vertices={bare['n_dofs'] == m.p.shape[1]}")
    print(f"bare_p1_upwind_flux_matrix_is_zero={bare['flux_max'] < 1e-14}")
    print(f"bare_p1_penalty_matrix_is_zero={bare['pen_max'] < 1e-14}")
    print(f"bare_p1_solution_has_no_jumps={bare['sol_jump_max'] < 1e-12}")
    print(f"bare_p1_flux_max={bare['flux_max']:.3e} bare_p1_pen_max={bare['pen_max']:.3e}")
    if bare["flux_max"] >= 1e-14 or bare["pen_max"] >= 1e-14:
        print("FAIL: the DG facet forms did NOT vanish on the bare element",
              file=sys.stderr)
        ok = False
    if bare["sol_jump_max"] >= 1e-12:
        print("FAIL: the bare-element solution was not continuous", file=sys.stderr)
        ok = False
    if bare["n_dofs"] != m.p.shape[1]:
        print("FAIL: bare ElementTriP1 did not give one DOF per vertex",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: ElementDG wrapper ---------------------------------
    wrapped = probe(m, ElementDG(ElementTriP1()))
    print(f"dg_n_dofs={wrapped['n_dofs']}")
    print(f"dg_dofs_equal_3_times_elements={wrapped['n_dofs'] == 3 * m.nelements}")
    print(f"dg_upwind_flux_matrix_nonzero={wrapped['flux_max'] > 1e-6}")
    print(f"dg_penalty_matrix_nonzero={wrapped['pen_max'] > 1e-6}")
    print(f"dg_solution_jumps_at_element_edges={wrapped['sol_jump_max'] > 1e-3}")
    print(f"dg_flux_max={wrapped['flux_max']:.3e} dg_sol_jump_max={wrapped['sol_jump_max']:.3e}")
    if not (wrapped["flux_max"] > 1e-6 and wrapped["pen_max"] > 1e-6):
        print("FAIL: the DG facet forms vanished on the WRAPPED element too",
              file=sys.stderr)
        ok = False
    if wrapped["sol_jump_max"] <= 1e-3:
        print("FAIL: the wrapped-element solution showed no inter-element jumps",
              file=sys.stderr)
        ok = False
    if wrapped["n_dofs"] != 3 * m.nelements:
        print("FAIL: ElementDG(ElementTriP1()) DOF count is not 3 per element",
              file=sys.stderr)
        ok = False

    print(f"dofs_ratio_dg_over_bare={wrapped['n_dofs'] / bare['n_dofs']:.4f}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
