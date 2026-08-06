"""Tier-2: what condense's D argument accepts, and which form is silently wrong.

Claim: skfem poisson#4 -- D takes a flat int array of DOF indices; for
non-homogeneous BCs use condense(K, f, D=D, x=g) with a full-length g. The
2026-08-03 correction states that passing the DofsView object itself is NOT
wrong on 12.0.1.

This fixture confirms the correction (bit-identical results) and then finds the
form that IS silently wrong and is not in the claim: a BOOLEAN MASK. numpy
booleans index as the integers 0 and 1, so condense constrains DOFs 0 and 1
instead of the boundary, returns without warning, and the answer is an order of
magnitude too large. An empty D leaves the pure Neumann system singular and
scipy hands back ~1e14 garbage.

Mutation control: T2_MUTATE=1 converts the boolean mask to the flat int array
of DOF indices that D actually wants (np.flatnonzero(mask)) before handing it
to condense -- the documented fix, applied at the argument the pitfall is
about. The boundary is then really constrained, the answer is correct, and
"boolean_mask_answer_is_wrong=True" plus
"boolean_mask_max_over_correct_max_gt_5=True" disappear from the output, so the
fixture goes red. Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, unit_load

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True
    basis = Basis(MeshTri().refined(2), ElementTriP1())
    K = laplace.assemble(basis)
    f = unit_load.assemble(basis)
    dofs = basis.get_dofs()
    flat = dofs.flatten()
    print(f"basis_N={basis.N}")
    print(f"n_constrained={len(flat)}")

    u_view = solve(*condense(K, f, D=dofs))
    u_flat = solve(*condense(K, f, D=flat))
    identical = bool(np.array_equal(u_view, u_flat))
    print(f"dofsview_and_flatten_bit_identical={identical}")
    if not identical:
        print("FAIL: DofsView and .flatten() disagree, so the 2026-08-03 "
              "correction is itself wrong", file=sys.stderr)
        ok = False
    reference = float(np.abs(u_flat).max())

    # --- WRONG variant (a): boolean mask --------------------------------
    mask = np.zeros(basis.N, dtype=bool)
    mask[flat] = True
    # THE PATHOLOGY: handing condense the boolean mask itself.  The documented
    # fix is to pass a flat int array of DOF indices instead.
    d_arg = mask if not MUTATE else np.flatnonzero(mask)
    raised = ""
    try:
        u_mask = solve(*condense(K, f, D=d_arg))
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
        u_mask = np.array([np.nan])
    print(f"boolean_mask_accepted_without_raising={not raised}")
    mask_max = float(np.abs(u_mask).max())
    wrong = np.isfinite(mask_max) and mask_max > 5.0 * reference
    print(f"boolean_mask_answer_is_wrong={wrong}")
    print(f"boolean_mask_max_over_correct_max_gt_5={wrong}")
    if raised or not wrong:
        print(f"FAIL: the boolean-mask path did not silently give a wrong "
              f"answer (raised={raised!r}, max {mask_max!r} vs correct "
              f"{reference!r})", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): empty D -------------------------------------
    u_empty = solve(*condense(K, f, D=np.array([], dtype=int)))
    huge = float(np.abs(u_empty).max()) > 1e6
    print(f"empty_d_gives_huge_garbage={huge}")
    if not huge:
        print(f"FAIL: the unconstrained system did not blow up: max "
              f"{np.abs(u_empty).max()!r}", file=sys.stderr)
        ok = False

    # --- RIGHT variant: non-homogeneous x --------------------------------
    x = basis.zeros()
    x[flat] = 3.5
    u_nh = solve(*condense(K, f, x=x, D=flat))
    prescribed = bool(np.allclose(u_nh[flat], 3.5, atol=1e-12))
    print(f"nonhomogeneous_x_reproduces_prescribed_values={prescribed}")
    if not prescribed:
        print(f"FAIL: prescribed values not reproduced: {u_nh[flat][:5]!r}",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
