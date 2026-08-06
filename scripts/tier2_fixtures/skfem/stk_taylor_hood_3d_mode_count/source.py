"""Tier-2: 3D Taylor-Hood, and the GridFunction that is not there.

Claim: skfem stokes#3 -- corrected 2026-08-06.  The entry previously said the
equal-order tet pair "produces a pressure GridFunction whose mode pattern
alternates by element (checkerboard); the Functional integral of div(u) over
the mesh is O(1) instead of O(eps)".  It was marked inherited and never
verified.

Measured on skfem 12.0.1:

  * `GridFunction` IS NGSOLVE VOCABULARY AND DOES NOT EXIST HERE.
    hasattr(skfem, 'GridFunction') is False, and no name containing
    "GridFunction" appears anywhere in the skfem namespace.  A skfem Stokes
    solution is a plain 1-D numpy.ndarray over the stacked [u; p] vector; the
    pressure is a slice at offset basis_u.N.  This is the same class of
    defect as a fabricated exception string, and it is checkable in one line.
  * ElementTetP2 and ElementTetP1 both exist, alongside the rest of the tet
    family, so the recommendation itself is sound.
  * THE INF-SUP FAILURE IS COUNTABLE WITHOUT SOLVING ANYTHING.  Assemble the
    block, condense the velocity boundary plus one pinned pressure DOF, and
    take the nullity: for Taylor-Hood it is CONSTANT across three refinement
    levels, while for equal-order P1/P1 it GROWS with the mesh.  That is a
    sharper and mesh-independent statement of "checkerboard", and it needs
    no eyeballing of a field.
  * a lightly regularised solve confirms it from the other side: the
    Taylor-Hood pressure range is bounded and converging under refinement
    while the equal-order range grows by an order of magnitude.
  * NOT REPRODUCED, and so not asserted: the entry's "integral of div(u) is
    O(1) instead of O(eps)".  Under a regularised solve the relative
    divergence of the two pairs is comparable and does not separate them, so
    that observable is not usable as written.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

import numpy as np
import scipy.sparse as sp
import skfem
from skfem import (
    Basis,
    BilinearForm,
    ElementTetP1,
    ElementTetP2,
    ElementVector,
    LinearForm,
    MeshTet,
    condense,
    solve,
)
from skfem.helpers import ddot, div, grad


@BilinearForm
def a_form(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def b_form(u, q, w):
    return -div(u) * q


@BilinearForm
def p_mass(p, q, w):
    return p * q


@LinearForm
def body(v, w):
    return np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1]) * v[2]


def block(vel, pres, refine):
    m = MeshTet().refined(refine)
    bu = Basis(m, vel, intorder=4)
    bp = Basis(m, pres, intorder=4)
    A = a_form.assemble(bu)
    B = b_form.assemble(bu, bp)
    return bu, bp, sp.bmat([[A, B.T], [B, None]], format="csr")


def spurious_modes(vel, pres, refine):
    bu, bp, K = block(vel, pres, refine)
    D = np.concatenate([bu.get_dofs().all(), [bu.N]])
    Kc, _, _, _ = condense(K, np.zeros(K.shape[0]), D=D)
    ev = np.linalg.eigvalsh(Kc.toarray())
    return int((np.abs(ev) < 1e-9).sum()), Kc.shape[0], bp.N


def pressure_range(vel, pres, refine, eps=1e-8):
    m = MeshTet().refined(refine)
    bu = Basis(m, vel, intorder=4)
    bp = Basis(m, pres, intorder=4)
    A = a_form.assemble(bu)
    B = b_form.assemble(bu, bp)
    Mp = p_mass.assemble(bp)
    K = sp.bmat([[A, B.T], [B, -eps * Mp]], format="csr")
    f = np.zeros(K.shape[0])
    f[:bu.N] = body.assemble(bu)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = solve(*condense(K, f, D=bu.get_dofs().all()))
        msgs = sorted({str(c.message) for c in caught})
    p = s[bu.N:]
    return float(p.max() - p.min()), msgs


def main() -> int:
    ok = True
    print(f"skfem_version={skfem.__version__}")

    # --- the foreign symbol -----------------------------------------------
    print(f"hasattr_skfem_GridFunction={hasattr(skfem, 'GridFunction')}")
    matches = [n for n in dir(skfem) if "gridfunction" in n.lower()]
    print(f"names_containing_gridfunction={matches}")
    print(f"gridfunction_is_foreign_vocabulary="
          f"{not hasattr(skfem, 'GridFunction') and not matches}")
    if hasattr(skfem, "GridFunction") or matches:
        print("FAIL: skfem does expose a GridFunction after all",
              file=sys.stderr)
        ok = False

    tets = sorted(n for n in dir(skfem) if n.startswith("ElementTet"))
    print(f"tet_elements={tets}")
    print(f"ElementTetP2_present={'ElementTetP2' in tets}")
    print(f"ElementTetP1_present={'ElementTetP1' in tets}")
    if not ("ElementTetP2" in tets and "ElementTetP1" in tets):
        print("FAIL: the recommended tet elements are missing",
              file=sys.stderr)
        ok = False

    # --- what a skfem solution actually is --------------------------------
    bu, bp, K = block(ElementVector(ElementTetP2()), ElementTetP1(), 1)
    D = np.concatenate([bu.get_dofs().all(), [bu.N]])
    f = np.zeros(K.shape[0])
    f[:bu.N] = body.assemble(bu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve(*condense(K, f, D=D))
    print(f"solution_type={type(sol).__name__}")
    print(f"solution_is_plain_ndarray={type(sol) is np.ndarray}")
    print(f"solution_ndim={sol.ndim}")
    print(f"solution_length_equals_block={len(sol) == K.shape[0]}")
    print(f"pressure_slice_offset_is_basis_u_N={bu.N}")
    print(f"pressure_slice_length={len(sol[bu.N:])} p_dofs={bp.N}")
    print(f"pressure_is_a_slice_at_offset={len(sol[bu.N:]) == bp.N}")
    if type(sol) is not np.ndarray or len(sol[bu.N:]) != bp.N:
        print("FAIL: the solution is not a flat ndarray sliced at basis_u.N",
              file=sys.stderr)
        ok = False

    # --- the mode count ----------------------------------------------------
    counts = {}
    for tag, vel, pres in (("th", ElementVector(ElementTetP2()),
                            ElementTetP1()),
                           ("eq", ElementVector(ElementTetP1()),
                            ElementTetP1())):
        series = []
        for r in (1, 2, 3):
            n, dim, npres = spurious_modes(vel, pres, r)
            series.append(n)
            print(f"{tag}_r{r}_condensed={dim} pressure_dofs={npres} "
                  f"spurious_modes={n}")
        counts[tag] = series
        print(f"{tag}_mode_counts={series}")

    th, eq = counts["th"], counts["eq"]
    print(f"taylor_hood_mode_count_is_constant={len(set(th)) == 1}")
    print(f"equal_order_mode_count_grows={eq[0] < eq[1] < eq[2]}")
    print(f"equal_order_exceeds_taylor_hood={min(eq) > max(th)}")
    print(f"mode_count_separates_the_pairs="
          f"{len(set(th)) == 1 and eq[0] < eq[1] < eq[2]}")
    if len(set(th)) != 1:
        print(f"FAIL: the Taylor-Hood mode count was not constant: {th}",
              file=sys.stderr)
        ok = False
    if not eq[0] < eq[1] < eq[2]:
        print(f"FAIL: the equal-order mode count did not grow: {eq}",
              file=sys.stderr)
        ok = False

    # --- and the pressure range --------------------------------------------
    ranges = {}
    for tag, vel, pres in (("th", ElementVector(ElementTetP2()),
                            ElementTetP1()),
                           ("eq", ElementVector(ElementTetP1()),
                            ElementTetP1())):
        vals = []
        for r in (2, 3):
            rng, msgs = pressure_range(vel, pres, r)
            vals.append(rng)
            print(f"{tag}_r{r}_pressure_range={rng:.4e} warnings={msgs!r}")
        ranges[tag] = vals
    print(f"taylor_hood_pressure_range_bounded="
          f"{ranges['th'][1] < 2.0 * ranges['th'][0]}")
    print(f"equal_order_pressure_range_grows_10x="
          f"{ranges['eq'][1] > 10.0 * ranges['eq'][0]}")
    if not ranges["th"][1] < 2.0 * ranges["th"][0]:
        print("FAIL: the Taylor-Hood pressure range was not bounded",
              file=sys.stderr)
        ok = False
    if not ranges["eq"][1] > 10.0 * ranges["eq"][0]:
        print("FAIL: the equal-order pressure range did not grow",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
