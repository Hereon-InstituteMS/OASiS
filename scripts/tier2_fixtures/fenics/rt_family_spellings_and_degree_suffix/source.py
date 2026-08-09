"""Tier-2 for fenics mixed_poisson#8: both spellings of Raviart-Thomas work and
build the SAME space -- basix.ufl.element("RT", domain.basix_cell(), k) and the
tuple shorthand fem.functionspace(domain, ("RT", k)). The names that really do
fail are the old DOLFIN degree-suffixed ones ("RT1", "P1"), which raise
ValueError "Unknown element family: ...", and ufl.FiniteElement, which no longer
exists on the ufl module at all.

The wrong variant is therefore writing the degree into the family name and
reaching for ufl.FiniteElement. The right one is the family name plus a separate
degree argument.

Observed on dolfinx 0.10.0 / basix 0.10.0: both RT spellings give 56 dofs on a
4x4 unit square and compare equal as elements; "RT1" raises
ValueError: Unknown element family: RT1 with cell type triangle; ufl.FiniteElement
raises AttributeError: module 'ufl' has no attribute 'FiniteElement'. Passing
"RT" to fem.functionspace raises nothing at all, so the AttributeError about
dolfinx.fem.FiniteElement quoted in older knowledge cannot be produced this way.

Mutation control: T2_MUTATE=1 uses the correct spellings in both places, so
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
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)

    V_expl = dolfinx.fem.functionspace(
        msh, basix.ufl.element("RT", msh.basix_cell(), 1))
    V_tuple = dolfinx.fem.functionspace(msh, ("RT", 1))
    same_n = ndofs(V_expl) == ndofs(V_tuple)
    same_el = V_expl.ufl_element() == V_tuple.ufl_element()
    print(f"rt_dofs_explicit={ndofs(V_expl)} rt_dofs_tuple={ndofs(V_tuple)}")
    print(f"rt_spellings_give_the_same_dof_count={same_n}")
    print(f"rt_spellings_give_the_same_element={same_el}")
    print(f"tuple_shorthand_raised_nothing={ndofs(V_tuple) > 0}")

    # 1) the family name with the degree glued on
    family = "RT" if MUTATE else "RT1"
    suffix_msg = ""
    try:
        dolfinx.fem.functionspace(msh, (family, 1))
        suffix_ok = True
    except ValueError as exc:
        suffix_ok = False
        suffix_msg = f"{type(exc).__name__}: {exc}"

    # 2) the legacy ufl element class
    legacy_msg = ""
    try:
        if MUTATE:
            basix.ufl.element("Lagrange", msh.basix_cell(), 1)
        else:
            ufl.FiniteElement("Lagrange", ufl.triangle, 1)
        legacy_ok = True
    except AttributeError as exc:
        legacy_ok = False
        legacy_msg = f"{type(exc).__name__}: {exc}"

    print(f"family_name_under_test={family!r} accepted={suffix_ok}")
    if suffix_msg:
        print(f"degree_suffix_error: {suffix_msg}")
    print(f"degree_suffixed_family_rejected={not suffix_ok}")
    if legacy_msg:
        print(f"legacy_class_error: {legacy_msg}")
    print(f"ufl_finiteelement_is_gone={not legacy_ok}")

    if same_n and same_el and not suffix_ok and not legacy_ok:
        print("VERDICT=both_rt_spellings_agree_only_the_legacy_names_fail")
        return 0
    print("VERDICT=legacy_spellings_were_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
