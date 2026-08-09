"""Tier-2 for fenics reaction_diffusion#3: a Dirichlet condition on ONE species
of a mixed space needs the collapsed sub-space in BOTH the dof lookup and the
value. The only combination that works is

    V0, _ = W.sub(0).collapse()
    dofs = fem.locate_dofs_topological((W.sub(0), V0), fdim, facets)
    g = fem.Function(V0)
    bc = fem.dirichletbc(g, dofs, W.sub(0))

Wrong variants tried here: a fem.Constant instead of a Function on V0; the dof
lookup without the (W.sub(0), V0) space pair; and dropping the trailing
W.sub(0).

Observed on dolfinx 0.10.0: all three raise the same
TypeError: __init__(): incompatible function arguments. The following argument
types are supported: ... and the last line names what was passed, e.g.
"Invoked with types: dolfinx.cpp.fem.DirichletBC_float64,
dolfinx.cpp.fem.Constant_float64, list, dolfinx.cpp.fem.FunctionSpace_float64".
The correct combination returns a DirichletBC whose dof set is exactly the
species-0 dofs on the marked facets.

Mutation control: T2_MUTATE=1 runs the correct combination in all three slots.
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
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    V0, sub_map = W.sub(0).collapse()
    facets = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_pair = dolfinx.fem.locate_dofs_topological(
        (W.sub(0), V0), fdim, facets)
    dofs_plain = dolfinx.fem.locate_dofs_topological(W.sub(0), fdim, facets)
    g = dolfinx.fem.Function(V0)
    g.x.array[:] = 0.5

    def good():
        return dolfinx.fem.dirichletbc(g, dofs_pair, W.sub(0))

    variants = {
        "constant_instead_of_function":
            (lambda: dolfinx.fem.dirichletbc(
                dolfinx.fem.Constant(msh, 0.5), dofs_pair, W.sub(0))),
        "dofs_without_the_space_pair":
            (lambda: dolfinx.fem.dirichletbc(g, dofs_plain, W.sub(0))),
        "trailing_subspace_omitted":
            (lambda: dolfinx.fem.dirichletbc(g, dofs_pair)),
    }

    failed = 0
    for name, wrong in variants.items():
        call = good if MUTATE else wrong
        try:
            call()
            print(f"{name}_raised=False")
        except TypeError as exc:
            failed += 1
            print(f"{name}_raised=True")
            print(f"--- {name} ---")
            print(f"{type(exc).__name__}: {exc}")
            print(f"--- end {name} ---")

    bc = good()
    n_bc = len(bc._cpp_object.dof_indices()[0])
    n_expected = len(dofs_pair[0])
    print(f"all_three_wrong_spellings_raise_typeerror={failed == 3}")
    print(f"correct_spelling_returns={type(bc).__name__} "
          f"bc_dofs={n_bc} located_dofs={n_expected}")
    print(f"correct_spelling_constrains_the_located_species_dofs="
          f"{n_bc == n_expected and n_bc == 9}")

    if failed == 3 and n_bc == n_expected and n_bc == 9:
        print("VERDICT=species_bc_needs_the_collapsed_space_everywhere")
        return 0
    print("VERDICT=a_wrong_species_bc_spelling_was_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
