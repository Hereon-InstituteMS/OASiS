"""Tier-2: the convection block needs a VECTOR basis, and no exception says so.

Claim: skfem navier_stokes#2 -- the convection term must be built on
ElementVector(ElementTriP2()), not a scalar ElementTriP2.  Do NOT expect an
AttributeError: CellBasis DOES have a `split` method in skfem 12.0.1
(AbstractBasis.split(x), which splits a SOLUTION VECTOR into per-component
(x, basis) pairs, not the basis itself), so "'CellBasis' has no attribute
'split'" is never emitted; calling it with no argument raises
TypeError 'AbstractBasis.split() missing 1 required positional argument: x',
which says nothing about the element type.  Check the SHAPE, not an exception.

Measured on skfem 12.0.1, MeshTri().refined(2):

  * every part of the corrected entry reproduces.  The basis class really is
    CellBasis, it really has `split`, the no-argument call raises exactly the
    quoted TypeError, and `split(x)` returns per-component (array, basis)
    pairs -- one per spatial dimension.
  * the shape check is the working guard: the vector basis has exactly dim
    times the scalar DOF count.
  * and the failure the entry warns about is silent and dimensional.  A
    convection block assembled on the SCALAR basis comes out square at the
    scalar size, so it is a perfectly valid matrix -- it simply cannot be
    stacked against the velocity block, and scipy.sparse.bmat is what
    finally objects, with a message about dimension mismatch that names
    neither the element nor the basis.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- the convection block that goes into the system is assembled on
ElementVector(ElementTriP2()) instead of the scalar ElementTriP2.  It is then
the right size, so 'scalar_block_is_dim_times_too_small=True' and
'stacking_scalar_block_raises=True' disappear from the output.  The split /
TypeError lines above are pathology-independent and are supposed to survive.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

MUTATE = os.environ.get("T2_MUTATE") == "1"

import numpy as np
import scipy.sparse as sp
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    MeshTri,
)
from skfem.helpers import ddot, dot, grad


@BilinearForm
def scalar_convection(u, v, w):
    return dot(w["wind"], grad(u)) * v


@BilinearForm
def vector_convection(u, v, w):
    return dot(dot(grad(u), w["wind"]), v)


@BilinearForm
def viscous(u, v, w):
    return ddot(grad(u), grad(v))


def main() -> int:
    ok = True
    m = MeshTri().refined(2)
    bs = Basis(m, ElementTriP2())
    bv = Basis(m, ElementVector(ElementTriP2()))
    dim = m.p.shape[0]
    print(f"basis_class={type(bs).__name__}")
    print(f"class_is_CellBasis={type(bs).__name__ == 'CellBasis'}")
    print(f"scalar_N={bs.N} vector_N={bv.N} dim={dim}")
    print(f"vector_N_is_dim_times_scalar={bv.N == dim * bs.N}")
    if bv.N != dim * bs.N:
        print("FAIL: the vector basis is not dim times the scalar one",
              file=sys.stderr)
        ok = False

    # --- the attribute that IS there --------------------------------------
    print(f"cellbasis_has_split={hasattr(bs, 'split')}")
    attr_err = None
    try:
        getattr(bs, "split")
    except AttributeError as e:
        attr_err = e
    print(f"getattr_split_raises_AttributeError={attr_err is not None}")
    print(f"quoted_attributeerror_text_present="
          f"{'has no attribute' in str(attr_err) if attr_err else False}")
    if not hasattr(bs, "split") or attr_err is not None:
        print("FAIL: CellBasis does NOT have split, so the AttributeError "
              "the entry rules out could fire after all", file=sys.stderr)
        ok = False

    type_err = None
    try:
        bs.split()
    except TypeError as e:
        type_err = e
    quoted = ("AbstractBasis.split() missing 1 required positional "
              "argument: 'x'")
    print(f"split_no_arg_raises_TypeError={isinstance(type_err, TypeError)}")
    print(f"split_no_arg_message={str(type_err)!r}")
    print(f"quoted_typeerror_matches={quoted in str(type_err)}")
    print(f"message_mentions_element_type="
          f"{'Element' in str(type_err) or 'element' in str(type_err)}")
    if quoted not in str(type_err):
        print("FAIL: the no-argument split did not raise the quoted TypeError",
              file=sys.stderr)
        ok = False

    parts = bv.split(bv.zeros())
    print(f"split_with_vector_returns={type(parts).__name__} len={len(parts)}")
    print(f"split_gives_one_part_per_dimension={len(parts) == dim}")
    print(f"split_part_is_a_pair={len(parts[0]) == 2}")
    print(f"split_part_array_length={len(parts[0][0])} scalar_N={bs.N}")
    print(f"split_parts_are_scalar_sized={len(parts[0][0]) == bs.N}")
    if len(parts) != dim:
        print("FAIL: split did not give one part per component",
              file=sys.stderr)
        ok = False

    # --- the silent, dimensional failure -----------------------------------
    wind_s = bs.zeros() + 1.0
    wind_v = np.stack([bv.interpolate(bv.zeros() + 1.0).value[0]] * dim)
    # THE PATHOLOGY: the convection block built on the SCALAR basis.  Under
    # T2_MUTATE the documented fix is applied at this one site -- the same
    # block is built on ElementVector(ElementTriP2()) instead.
    if MUTATE:
        C_scalar = vector_convection.assemble(bv, wind=wind_v)
    else:
        C_scalar = scalar_convection.assemble(
            bs, wind=np.stack([bs.interpolate(wind_s)] * dim))
    C_vector = vector_convection.assemble(bv, wind=wind_v)
    A = viscous.assemble(bv)
    print(f"scalar_convection_shape={C_scalar.shape}")
    print(f"vector_convection_shape={C_vector.shape}")
    print(f"viscous_block_shape={A.shape}")
    print(f"scalar_block_is_square_and_valid="
          f"{C_scalar.shape[0] == C_scalar.shape[1] and C_scalar.nnz > 0}")
    print(f"scalar_block_is_dim_times_too_small="
          f"{C_scalar.shape[0] * dim == A.shape[0]}")
    print(f"vector_block_matches_viscous_block="
          f"{C_vector.shape == A.shape}")
    if C_scalar.shape[0] * dim != A.shape[0]:
        print("FAIL: the scalar convection block was not dim times too small",
              file=sys.stderr)
        ok = False

    # matching intorder, or the pair assembly trips the quadrature guard
    bv4 = Basis(m, ElementVector(ElementTriP2()), intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)

    @BilinearForm
    def divergence(u, q, w):
        from skfem.helpers import div
        return -div(u) * q

    B = divergence.assemble(bv4, bp)
    stack_err = None
    try:
        sp.bmat([[A + C_scalar, B.T], [B, None]], format="csr")
    except Exception as e:                            # noqa: BLE001
        stack_err = e
    print(f"stacking_scalar_block_raises={stack_err is not None}")
    print(f"stacking_exception_type="
          f"{type(stack_err).__name__ if stack_err else None}")
    print(f"stacking_message={str(stack_err)[:80]!r}")
    print(f"stacking_message_names_the_basis="
          f"{'basis' in str(stack_err).lower()}")
    print(f"stacking_message_names_the_element="
          f"{'element' in str(stack_err).lower()}")
    if stack_err is None:
        print("FAIL: stacking the wrong-sized block did not fail at all",
              file=sys.stderr)
        ok = False
    if "basis" in str(stack_err).lower() or "element" in str(
            stack_err).lower():
        print("FAIL: the stacking error DOES name the basis or element, so "
              "it is not the uninformative failure documented here",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
