"""Tier-2 for fenics reaction_diffusion#0: the mixed space for a multi-species
system is built with basix.ufl.mixed_element and a LIST of elements, then
fem.functionspace.

Wrong variants: basix.ufl.mixed_element(P1, P1) with the elements as separate
positional arguments, and the legacy ufl.MixedElement([P1, P1]).

Observed on dolfinx 0.10.0 / basix 0.10.0:
  basix.ufl.mixed_element(P1, P1) -> TypeError: mixed_element() takes 1
      positional argument but 2 were given
  ufl.MixedElement([P1, P1])      -> AttributeError: module 'ufl' has no
      attribute 'MixedElement'
and the working call basix.ufl.mixed_element([P1, P1]) gives a 2-subspace
element. Species may use different degrees -- mixed_element([P1, P2]) is also a
valid 2-subspace element -- and when all species share one element the blocked
space fem.functionspace(msh, ("Lagrange", 1, (2,))) has the same global dof
count as the mixed one (162 on an 8x8 unit square).

Mutation control: T2_MUTATE=1 passes the list and uses basix instead of ufl, so
neither exception happens.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def ndofs(V) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2)

    # 1) elements as separate positional arguments
    pos_msg = ""
    try:
        if MUTATE:
            basix.ufl.mixed_element([P1, P1])
        else:
            basix.ufl.mixed_element(P1, P1)
        pos_ok = True
    except TypeError as exc:
        pos_ok = False
        pos_msg = f"{type(exc).__name__}: {exc}"

    # 2) the legacy ufl class
    legacy_msg = ""
    try:
        if MUTATE:
            basix.ufl.mixed_element([P1, P1])
        else:
            ufl.MixedElement([P1, P1])
        legacy_ok = True
    except AttributeError as exc:
        legacy_ok = False
        legacy_msg = f"{type(exc).__name__}: {exc}"

    if pos_msg:
        print(f"positional_call_error: {pos_msg}")
    print(f"mixed_element_rejects_positional_elements={not pos_ok}")
    if legacy_msg:
        print(f"legacy_call_error: {legacy_msg}")
    print(f"ufl_mixedelement_is_gone={not legacy_ok}")

    same = basix.ufl.mixed_element([P1, P1])
    mixdeg = basix.ufl.mixed_element([P1, P2])
    W = dolfinx.fem.functionspace(msh, same)
    Wb = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (2,)))
    print(f"mixed_element_list_subspaces={len(same.sub_elements)} "
          f"mixed_degrees_subspaces={len(mixdeg.sub_elements)}")
    print(f"mixed_dofs={ndofs(W)} blocked_dofs={ndofs(Wb)}")
    print(f"mixed_element_list_builds_two_subspaces="
          f"{len(same.sub_elements) == 2}")
    print(f"different_degrees_per_species_allowed="
          f"{len(mixdeg.sub_elements) == 2}")
    print(f"blocked_space_has_the_same_dof_count={ndofs(W) == ndofs(Wb)}")

    if (not pos_ok and not legacy_ok and len(same.sub_elements) == 2
            and len(mixdeg.sub_elements) == 2 and ndofs(W) == ndofs(Wb)):
        print("VERDICT=mixed_element_takes_one_list_argument")
        return 0
    print("VERDICT=positional_or_legacy_spelling_was_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
