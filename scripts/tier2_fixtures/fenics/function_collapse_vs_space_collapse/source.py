"""Tier-2 for fenics reaction_diffusion#2: fem.Function.collapse() returns ONE
Function; fem.FunctionSpace.collapse() returns a (FunctionSpace, dofmap) pair.
Unpacking the Function version into two names is the mistake.

Wrong variant: a_h, dofs = w.sub(0).collapse().
Right variants: a_h = w.sub(0).collapse() (a Function on the collapsed space,
ready for .x.array.min()) and V0, sub_map = W.sub(0).collapse() (the space plus
the index array, so w.x.array[sub_map] is that species' block).

Observed on dolfinx 0.10.0: the unpacking raises
NotImplementedError: Cannot take length of non-vector expression.
because Python calls len() on the returned Function. The two correct forms give
a dolfinx Function and a (FunctionSpace, ndarray) pair, and
w.x.array[sub_map] agrees with the collapsed Function's array exactly.

Mutation control: T2_MUTATE=1 assigns the result to a single name, and nothing
is raised.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    w = dolfinx.fem.Function(W)
    w.x.array[:] = np.arange(w.x.array.size, dtype=w.x.array.dtype)

    unpack_msg = ""
    try:
        if MUTATE:
            a_h = w.sub(0).collapse()
        else:
            a_h, dofs = w.sub(0).collapse()
        unpack_ok = True
    except NotImplementedError as exc:
        unpack_ok = False
        unpack_msg = f"{type(exc).__name__}: {exc}"

    if unpack_msg:
        print(f"unpack_error: {unpack_msg}")
    print(f"function_collapse_cannot_be_unpacked={not unpack_ok}")

    a_h = w.sub(0).collapse()
    pair = W.sub(0).collapse()
    V0, sub_map = pair
    print(f"function_collapse_returns={type(a_h).__name__} "
          f"space_collapse_returns=({type(pair[0]).__name__}, "
          f"{type(pair[1]).__name__}) len={len(pair)}")
    print(f"function_collapse_returns_one_function="
          f"{isinstance(a_h, dolfinx.fem.Function)}")
    print(f"space_collapse_returns_a_pair="
          f"{len(pair) == 2 and isinstance(V0, dolfinx.fem.FunctionSpace)}")
    block_matches = bool(np.array_equal(w.x.array[sub_map], a_h.x.array))
    print(f"species_min_from_collapsed_function={a_h.x.array.min():.1f}")
    print(f"sub_map_indexes_the_same_block={block_matches}")

    if (not unpack_ok and isinstance(a_h, dolfinx.fem.Function)
            and len(pair) == 2 and block_matches):
        print("VERDICT=function_collapse_is_not_a_pair")
        return 0
    print("VERDICT=function_collapse_unpacked_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
