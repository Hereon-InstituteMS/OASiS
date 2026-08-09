"""Tier-2: InteriorFacetBasis is single-sided, and skfem.helpers.jump silently
becomes the IDENTITY when a FEniCSx jump() form is ported verbatim.

Claim: skfem dg_methods#4 -- "scikit-fem uses a SINGLE-SIDED
InteriorFacetBasis -- '+' and '-' sides are implicit (assembly visits each
interior facet twice with sign-flipped normals, not once with an explicit
jump).  Porting a FEniCSx-style form that uses 'jump(u)' literally produces the
wrong factor of 2 -- in scikit-fem the jump appears naturally as
(u - u.other)."

Measured on skfem 12.0.1, MeshTri().refined(2) (32 elements, 56 facets,
16 boundary, 40 interior):
  TRUE  -- InteriorFacetBasis IS single-sided: one basis object carries one
           side, selected by ``side=`` (side=0 -> mesh.f2t[0], side=1 ->
           mesh.f2t[1]); the jump is formed by the 2x2 block assembly
           ``asm(form, [i0, i1], [i0, i1])``.
  FALSE -- it does NOT visit each interior facet twice: nelems = 40 = the
           number of interior facets, i.e. each facet exactly once.
  FALSE -- ``u.other`` does not exist.
  FALSE -- the verbatim port does not give a factor of 2.  ``jump(w, u, v)``
           returns its arguments UNCHANGED when ``w`` has no ``idx``, which is
           exactly what a single-basis ``form.assemble(i0)`` produces.  The
           operator then is not a rescaled jump -- it stops being a jump: on a
           CONTINUOUS field, where the true penalty must vanish, it returns
           O(1) instead of 0, and it carries zero inter-element coupling.

Wrong variant: assemble the FEniCSx-style jump(u)*jump(v) penalty on a single
InteriorFacetBasis and compare against the 2x2 block spelling.

Mutation control: ``T2_MUTATE=1 python source.py`` applies the documented fix at
the pathology site -- ``P_wrong`` is built with the 2x2 block spelling
``asm(penalty, [i0, i1], [i0, i1])`` instead of the single-basis
``penalty.assemble(i0)``.  The verbatim-port operator then IS the correct jump
penalty, so the pathological expectations vanish and the fixture goes red.
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
    InteriorFacetBasis,
    MeshTri,
    asm,
)
from skfem.helpers import jump

MUTATE = os.environ.get("T2_MUTATE") == "1"

IDX_SEEN: list = []


@BilinearForm
def penalty(u, v, w):
    """Verbatim FEniCSx-style interior penalty: jump(u) * jump(v) / h."""
    ju, jv = jump(w, u, v)
    return ju * jv / w.h


@BilinearForm
def probe_idx(u, v, w):
    IDX_SEEN.append(getattr(w, "idx", None))
    return 0.0 * u * v


@BilinearForm
def touch_other(u, v, w):
    return (u - u.other) * v


def main() -> int:
    ok = True
    m = MeshTri().refined(2)
    e = ElementDG(ElementTriP1())
    i0 = InteriorFacetBasis(m, e, side=0)
    i1 = InteriorFacetBasis(m, e, side=1)
    ib = Basis(m, e)

    n_bnd = int(np.sum(m.f2t[1] == -1))
    n_int = int(np.sum(m.f2t[1] != -1))
    print(f"n_elements={m.nelements}")
    print(f"n_facets={int(m.facets.shape[1])}")
    print(f"n_boundary_facets={n_bnd}")
    print(f"n_interior_facets={n_int}")
    print(f"interior_facet_basis_nelems={i0.nelems}")
    print(f"each_interior_facet_visited_once={i0.nelems == n_int}")
    print(f"visits_each_facet_twice={i0.nelems == 2 * n_int}")
    print(f"side0_tind_is_f2t0={bool(np.array_equal(i0.tind, m.f2t[0, i0.find]))}")
    print(f"side1_tind_is_f2t1={bool(np.array_equal(i1.tind, m.f2t[1, i1.find]))}")
    if i0.nelems != n_int:
        print("FAIL: InteriorFacetBasis no longer covers each interior facet once",
              file=sys.stderr)
        ok = False
    if not np.array_equal(i0.tind, m.f2t[0, i0.find]):
        print("FAIL: side=0 no longer selects mesh.f2t[0]", file=sys.stderr)
        ok = False

    # --- the catalog's (u - u.other) spelling does not exist ---------------
    try:
        asm(touch_other, [i0, i1], [i0, i1])
    except AttributeError as exc:
        other_msg = str(exc)
    else:
        other_msg = ""
    print(f"u_dot_other_exists={not other_msg}")
    print(f"u_dot_other_message={other_msg!r}")
    if "'DiscreteField' object has no attribute 'other'" not in other_msg:
        print("FAIL: u.other behaviour changed", file=sys.stderr)
        ok = False

    # --- WRONG variant: verbatim port assembled on ONE basis --------------
    IDX_SEEN.clear()
    probe_idx.assemble(i0)
    single_idx = IDX_SEEN[-1]
    IDX_SEEN.clear()
    asm(probe_idx, [i0, i1], [i0, i1])
    block_idx = IDX_SEEN[-1]
    print(f"single_basis_w_has_idx={single_idx is not None}")
    print(f"block_assembly_w_has_idx={block_idx is not None}")
    print(f"block_assembly_last_idx={block_idx}")
    if single_idx is not None:
        print("FAIL: a single-basis assemble now supplies w.idx", file=sys.stderr)
        ok = False
    if block_idx is None:
        print("FAIL: the 2x2 block assembly no longer supplies w.idx",
              file=sys.stderr)
        ok = False

    # verbatim FEniCSx port; under T2_MUTATE the documented fix (the 2x2 block
    # spelling) is applied at this very site instead.
    P_wrong = (penalty.assemble(i0) if not MUTATE
               else asm(penalty, [i0, i1], [i0, i1]))
    P_right = asm(penalty, [i0, i1], [i0, i1])   # the skfem spelling

    # A CONTINUOUS field: a true jump penalty must annihilate it.
    cont = ib.project(lambda x: 1.0 + x[0] + 2.0 * x[1])
    q_wrong = float(cont @ (P_wrong @ cont))
    q_right = float(cont @ (P_right @ cont))
    print(f"continuous_field_right_penalty_is_zero={abs(q_right) < 1e-10}")
    print(f"continuous_field_wrong_penalty_is_zero={abs(q_wrong) < 1e-10}")
    print(f"continuous_field_wrong_penalty_gt_0p1={abs(q_wrong) > 0.1}")
    print(f"q_right={q_right:.4e} q_wrong={q_wrong:.4e}")
    if abs(q_right) >= 1e-10:
        print("FAIL: the 2x2 block jump penalty did not vanish on a continuous field",
              file=sys.stderr)
        ok = False
    if abs(q_wrong) < 1e-10:
        print("FAIL: the verbatim single-basis port behaved like a real jump",
              file=sys.stderr)
        ok = False

    # The error is NOT a factor of two: no scalar makes the two agree.
    rng = np.random.default_rng(0)
    z = rng.standard_normal(ib.N)
    ratio = float(z @ (P_wrong @ z)) / float(z @ (P_right @ z))
    resid = float(abs(P_wrong - ratio * P_right).max())
    print(f"wrong_over_right_ratio_is_2={abs(ratio - 2.0) < 0.05}")
    print(f"wrong_over_right_ratio_is_0p5={abs(ratio - 0.5) < 0.05}")
    print(f"wrong_is_not_a_scalar_multiple_of_right={resid > 1e-8}")
    print(f"wrong_over_right_ratio={ratio:.6f} residual={resid:.4e}")
    if resid <= 1e-8:
        print("FAIL: the verbatim port WAS a pure rescaling of the correct operator",
              file=sys.stderr)
        ok = False

    # The verbatim port also loses the inter-element coupling entirely.
    dof2el = np.empty(ib.N, dtype=int)
    for k in range(m.nelements):
        dof2el[ib.element_dofs[:, k]] = k

    def cross(M):
        C = M.tocoo()
        return int(np.sum((dof2el[C.row] != dof2el[C.col])
                          & (np.abs(C.data) > 1e-14)))

    print(f"wrong_interelement_entries={cross(P_wrong)}")
    print(f"right_interelement_entries_gt_0={cross(P_right) > 0}")
    if cross(P_wrong) != 0:
        print("FAIL: the single-basis port unexpectedly coupled elements",
              file=sys.stderr)
        ok = False
    if cross(P_right) <= 0:
        print("FAIL: the 2x2 block penalty did not couple elements", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
