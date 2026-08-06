"""Tier-2: the legacy MeshQuad((nx, ny)) call does NOT raise -- it builds junk.

Claim: skfem wave#1 -- skfem >= 12 expects MeshQuad.init_tensor for
tensor-product Q1 grids; the older MeshQuad((nx, ny)) constructor was removed.
"Signal: TypeError 'MeshQuad() takes 1 positional argument but 2 were given'
or AttributeError 'type object MeshQuad has no attribute __init__' on the mesh
construction line."

Measured on skfem 12.0.1: BOTH quoted signals are absent, and the real
behaviour is worse than either.

  * MeshQuad((10, 10)) RAISES NOTHING.  skfem 12's Mesh is a dataclass whose
    first field is `doflocs`, so the tuple is taken as coordinates: the call
    returns a MeshQuad1 whose point array is the 1-D array [10., 10.] and
    whose connectivity is a single element referencing nodes 0..3.  A mesh
    object exists, `type(...).__name__` is MeshQuad1, and an `except` around
    the construction line never fires.
  * the only thing emitted at construction is a bare line of text on stdout,
    "Unable to calculate global DOF locations." -- not a warning, not an
    exception.
  * the failure surfaces LATER, at Basis(...), as
    IndexError: too many indices for array: array is 1-dimensional, but 2
    were indexed -- a message that says nothing about the constructor, on a
    line the user did not suspect.
  * MeshQuad(10, 10) (two positional ints rather than a tuple) raises
    IndexError, not the quoted TypeError either.
  * init_tensor is the working call and produces the expected element count.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import skfem
from skfem import Basis, ElementQuad1, MeshQuad


def main() -> int:
    ok = True
    print(f"skfem_version={skfem.__version__}")

    # --- the legacy call ------------------------------------------------
    err = None
    obj = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            obj = MeshQuad((10, 10))
        except Exception as e:                       # noqa: BLE001
            err = e
        msgs = sorted({str(c.message) for c in caught})
    print(f"legacy_tuple_call_raises={err is not None}")
    print(f"legacy_tuple_call_exception={type(err).__name__ if err else None}")
    print(f"legacy_tuple_call_warnings={msgs!r}")
    if err is not None:
        print(f"FAIL: MeshQuad((10, 10)) raised {type(err).__name__}: {err}",
              file=sys.stderr)
        ok = False
    else:
        print(f"legacy_result_class={type(obj).__name__}")
        print(f"legacy_result_p_shape={obj.p.shape}")
        print(f"legacy_result_t_shape={obj.t.shape}")
        print(f"legacy_result_p_values={obj.p.tolist()}")
        print(f"legacy_p_is_one_dimensional={obj.p.ndim == 1}")
        print(f"legacy_has_a_single_bogus_element={obj.t.shape[1] == 1}")
        if obj.p.ndim != 1 or obj.t.shape[1] != 1:
            print("FAIL: the legacy call did not build the degenerate mesh "
                  "documented here", file=sys.stderr)
            ok = False

        # --- where the damage actually shows up ------------------------
        later = None
        try:
            Basis(obj, ElementQuad1())
        except Exception as e:                       # noqa: BLE001
            later = e
        print(f"basis_on_legacy_mesh_raises={later is not None}")
        print(f"basis_exception_type={type(later).__name__ if later else None}")
        print(f"basis_exception_message={str(later)!r}")
        print(f"basis_message_mentions_MeshQuad="
              f"{'MeshQuad' in str(later)}")
        if later is None:
            print("FAIL: the degenerate mesh built a Basis without error",
                  file=sys.stderr)
            ok = False

    # --- the quoted strings are not produced ---------------------------
    texts = []
    for call, label in (((10, 10), "tuple"), ):
        try:
            MeshQuad(call)
        except Exception as e:                       # noqa: BLE001
            texts.append(str(e))
    two_arg = None
    try:
        MeshQuad(10, 10)
    except Exception as e:                           # noqa: BLE001
        two_arg = e
    texts.append(str(two_arg))
    print(f"two_positional_ints_exception="
          f"{type(two_arg).__name__ if two_arg else None}")
    print(f"two_positional_ints_message={str(two_arg)!r}")
    blob = " ".join(texts)
    q1 = "takes 1 positional argument but 2 were given"
    q2 = "has no attribute '__init__'"
    print(f"quoted_typeerror_text_present={q1 in blob}")
    print(f"quoted_attributeerror_text_present={q2 in blob}")
    if q1 in blob or q2 in blob:
        print("FAIL: one of the quoted messages WAS produced",
              file=sys.stderr)
        ok = False

    # --- the working call -----------------------------------------------
    good = MeshQuad.init_tensor(np.linspace(0.0, 1.0, 11),
                                np.linspace(0.0, 1.0, 11))
    ib = Basis(good, ElementQuad1())
    print(f"init_tensor_class={type(good).__name__}")
    print(f"init_tensor_elements={good.t.shape[1]}")
    print(f"init_tensor_vertices={good.p.shape[1]}")
    print(f"init_tensor_basis_N={ib.N}")
    print(f"init_tensor_gives_100_elements={good.t.shape[1] == 100}")
    if good.t.shape[1] != 100 or ib.N != 121:
        print("FAIL: init_tensor did not build the expected 10x10 grid",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
