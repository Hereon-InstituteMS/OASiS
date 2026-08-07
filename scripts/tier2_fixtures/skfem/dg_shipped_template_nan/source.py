"""Tier-2: the shipped dg_methods_2d template returns NaN everywhere and exits rc=0.

Claim: skfem dg_methods#6 -- running the shipped ``dg_methods_2d`` template
unmodified on skfem 12.0.1 prints scipy's ``MatrixRankWarning: Matrix is
exactly singular`` at the ``spsolve`` line, then ``max(u) = nan, min(u) = nan``
with 6144 DOFs / 2048 elements, and still returns 0.  The pure-advection DG
operator has no inflow boundary term pinning the solution, so it is
rank-deficient.  Remedy given by the catalog: add the inflow FacetBasis term
(or a small reaction/mass shift) and re-check that the matrix is non-singular.

Wrong variant: this fixture rebuilds the shipped operator verbatim (the same
``advection_volume`` + ``outflow_flux`` on FacetBasis + ``upwind_interior``
on a single InteriorFacetBasis, no inflow bilinear term) on the template's own
default mesh (nx=32 MeshQuad -> to_meshtri) and solves it with spsolve.

Observed on skfem 12.0.1 / scipy 1.15.3:
  * spsolve emits ``MatrixRankWarning: Matrix is exactly singular``
  * every entry of u is NaN, yet the script would exit 0
  * rank deficiency is exactly 3 (one DG-P1 element's worth of DOFs)
  * the shipped interior-facet term contributes ZERO inter-element coupling,
    because it is assembled on ONE InteriorFacetBasis instead of the
    side=0 / side=1 pair -- so adding the inflow term removes the singularity
    but does NOT make the answer right; the interior flux must be fixed too.

Mutation control: T2_MUTATE=1 turns every fix="none" build into fix="inflow",
i.e. the catalog's own remedy (add the inflow FacetBasis term) applied inside
build() where the shipped operator is assembled. The shipped leg then solves,
so "Matrix is exactly singular", shipped_all_nan=True, shipped_any_finite=False
and rank_deficiency=3 all vanish and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementDG,
    ElementTriP1,
    FacetBasis,
    InteriorFacetBasis,
    LinearForm,
    MeshQuad,
    asm,
)
from scipy.sparse.linalg import spsolve

MUTATE = os.environ.get("T2_MUTATE") == "1"

B = np.array([1.0, 0.5])
BB = float(B @ B)


def make_mesh(nx: int):
    # exactly what the shipped template does
    return MeshQuad.init_tensor(
        np.linspace(0, 1, nx + 1),
        np.linspace(0, 1, nx + 1),
    ).to_meshtri()


# ----- forms copied from the shipped template ---------------------------
@BilinearForm
def advection_volume(u, v, w):
    return (B[0] * u.grad[0] + B[1] * u.grad[1]) * v


@BilinearForm
def upwind_interior(u, v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    return bn * u * v


@BilinearForm
def outflow_flux(u, v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    return np.where(bn > 0, bn, 0.0) * u * v


@LinearForm
def source(v, w):
    return 1.0 * v


# ----- the term the shipped template is missing --------------------------
@BilinearForm
def inflow_bilinear(u, v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    return -np.minimum(bn, 0.0) * u * v


@BilinearForm
def mass(u, v, w):
    return u * v


# ----- a correctly assembled upwind DG operator --------------------------
@BilinearForm
def upwind_interior_2x2(u, v, w):
    """Single-sided upwind flux, assembled over the side=0/side=1 pair."""
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    sv = (-1.0) ** w.idx[1]          # +1 when v lives on side 0
    ju = (-1.0) ** w.idx[0] * u      # contributes to the jump u0 - u1
    return np.minimum(sv * bn, 0.0) * (-sv) * ju * v


def build(nx: int, fix: str = "none"):
    # Pathology: the shipped operator carries no inflow boundary term, so it is
    # rank-deficient.  Mutation: apply the catalog's remedy at that site.
    if fix == "none" and MUTATE:
        fix = "inflow"
    m = make_mesh(nx)
    e = ElementDG(ElementTriP1())
    ib = Basis(m, e)
    ibf = FacetBasis(m, e)
    ifi = InteriorFacetBasis(m, e)

    A = asm(advection_volume, ib) + asm(outflow_flux, ibf) + asm(upwind_interior, ifi)
    if fix == "inflow":
        A = A + asm(inflow_bilinear, ibf)
    elif fix == "shift":
        A = A + 1e-2 * asm(mass, ib)
    elif fix == "full":
        i0 = InteriorFacetBasis(m, e, side=0)
        i1 = InteriorFacetBasis(m, e, side=1)
        A = (asm(advection_volume, ib)
             + asm(upwind_interior_2x2, [i0, i1], [i0, i1])
             + asm(inflow_bilinear, ibf))
    f = asm(source, ib)
    return A, f, ib, m


def solve_catching(A, f):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        u = spsolve(A.tocsr(), f)
        msgs = [str(c.message) for c in caught]
    return u, msgs


def main() -> int:
    ok = True

    # --- WRONG variant: the shipped operator, template default mesh -------
    A, f, ib, m = build(32, fix="none")
    u, msgs = solve_catching(A, f)
    print(f"n_dofs={A.shape[0]}")
    print(f"n_elements={m.nelements}")
    print(f"shipped_all_nan={bool(np.isnan(u).all())}")
    print(f"shipped_any_finite={bool(np.isfinite(u).any())}")
    print(f"scipy_warning={msgs!r}")
    print("rc_only_gate_would_pass=True")     # spsolve warns, it does not raise
    if not np.isnan(u).all():
        print("FAIL: the shipped DG operator did not produce an all-NaN solution",
              file=sys.stderr)
        ok = False
    if not any("Matrix is exactly singular" in s for s in msgs):
        print("FAIL: scipy did not report an exactly singular matrix",
              file=sys.stderr)
        ok = False

    # --- why: rank deficiency + no inter-element coupling ------------------
    A8, f8, ib8, m8 = build(8, fix="none")
    dense = A8.toarray()
    rank = int(np.linalg.matrix_rank(dense))
    print(f"small_n_dofs={A8.shape[0]}")
    print(f"rank_deficiency={A8.shape[0] - rank}")
    if A8.shape[0] - rank < 1:
        print("FAIL: the shipped operator was full rank", file=sys.stderr)
        ok = False

    e8 = ElementDG(ElementTriP1())
    ibc = Basis(m8, e8)
    dof2el = np.empty(ibc.N, dtype=int)
    for k in range(m8.nelements):
        dof2el[ibc.element_dofs[:, k]] = k
    F = asm(upwind_interior, InteriorFacetBasis(m8, e8)).tocoo()
    n_cross = int(np.sum((dof2el[F.row] != dof2el[F.col])
                         & (np.abs(F.data) > 1e-14)))
    print(f"shipped_interior_term_interelement_entries={n_cross}")
    if n_cross != 0:
        print("FAIL: shipped single-basis interior term unexpectedly couples elements",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant A: the documented remedies remove the singularity ---
    for tag, fix in (("inflow_facetbasis", "inflow"), ("mass_shift", "shift")):
        Af, ff, _, _ = build(32, fix=fix)
        uf, mf = solve_catching(Af, ff)
        finite = bool(np.isfinite(uf).all())
        print(f"fix_{tag}_finite={finite}")
        print(f"fix_{tag}_singular_warning={bool(mf)}")
        if not finite or mf:
            print(f"FAIL: remedy '{tag}' did not give a non-singular solve",
                  file=sys.stderr)
            ok = False

    # --- RIGHT variant B: correct DG is finite AND physically sensible -----
    # f = 1 with zero inflow data on x=0 and y=0.  Along a streamline
    # x(t) = x0 + b t one has du/dt = b.grad(u) = 1, so u is the time since
    # entering the domain: u_exact = min(x, y / 0.5) = min(x, 2y), which lies
    # in [0, 1] and increases monotonically along the streamwise coordinate.
    Ac, fc, ibc2, mc = build(32, fix="full")
    uc, mc_msgs = solve_catching(Ac, fc)
    finite = bool(np.isfinite(uc).all())
    x = ibc2.doflocs
    uex = np.minimum(x[0] / B[0], x[1] / B[1])
    rel = float(np.linalg.norm(uc - uex) / np.linalg.norm(uex))
    s = B[0] * x[0] + B[1] * x[1]
    edges = np.linspace(s.min(), s.max(), 13)
    means = np.array([uc[(s >= edges[k]) & (s <= edges[k + 1])].mean()
                      for k in range(12)])
    n_dec = int(np.sum(np.diff(means) < -1e-9))
    sensible = bool(finite and uc.min() > -1e-2 and uc.max() < 1.05
                    and rel < 0.2 and n_dec == 0)
    print(f"correct_dg_finite={finite}")
    print(f"correct_dg_streamwise_decreases={n_dec}")
    print(f"correct_dg_rel_l2_lt_0p2={bool(rel < 0.2)}")
    print(f"correct_dg_physically_sensible={sensible}")
    print(f"correct_dg_min={uc.min():.6f} correct_dg_max={uc.max():.6f}")
    if not sensible:
        print("FAIL: the correctly assembled upwind DG operator was not sensible",
              file=sys.stderr)
        ok = False

    # --- the extra finding: 'add the inflow term' is necessary, not sufficient
    Ai, fi, ibi, _ = build(32, fix="inflow")
    ui, _ = solve_catching(Ai, fi)
    print(f"inflow_only_fix_min_is_negative={bool(ui.min() < -1e-3)}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
