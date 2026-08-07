"""Tier-2 for fenics stokes_darcy#2: a form that mixes a submesh Function with a
parent-mesh measure must be compiled with entity_maps, and in 0.10 that argument
is a SEQUENCE of EntityMap objects.

Wrong variants: (a) omit entity_maps entirely, (b) pass the older dict form
entity_maps={submesh: index_array}. Right variant: entity_maps=[cell_map] with
the EntityMap that create_submesh returned.

Observed on dolfinx 0.10.0: omitting the argument raises
"RuntimeError: Incompatible mesh. argument entity_maps must be provided."; the
dict form raises "TypeError: __init__(): incompatible function arguments." whose
printed overload list contains
"entity_maps: collections.abc.Sequence[dolfinx.cpp.mesh.EntityMap]". With
entity_maps=[cell_map] the form compiles and assembles, and so does the
rectangular coupling block between a submesh trial function and a parent test
function (81 x 45 on this 8x8 mesh). The silent trap reproduces too: integrating
the same integrand over the WHOLE parent domain gives exactly the sub-region
value, with no warning.

Mutation control: T2_MUTATE=1 compiles with entity_maps=[cell_map] from the
start, so neither error text appears.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import fem, mesh  # noqa: E402


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = msh.topology.dim
    por = mesh.locate_entities(msh, tdim, lambda x: x[1] <= 0.5 + 1e-12)
    sub, cell_map, _vmap, _gmap = mesh.create_submesh(msh, tdim, por)

    ids = np.full(msh.topology.index_map(tdim).size_local, 1, dtype=np.int32)
    ids[por] = 2
    ct = mesh.meshtags(msh, tdim, np.arange(ids.size, dtype=np.int32), ids)
    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)

    Vp = fem.functionspace(msh, ("Lagrange", 1))
    Vs = fem.functionspace(sub, ("Lagrange", 1))
    p_sub = fem.Function(Vs)
    p_sub.x.array[:] = 2.0
    v_par = ufl.TestFunction(Vp)
    integrand = p_sub * v_par

    no_map_err = ""
    dict_err = ""
    if not MUTATE:
        try:
            fem.form(integrand * dx(2))
            print("form_without_entity_maps_compiled=True")
        except RuntimeError as exc:
            no_map_err = f"{type(exc).__name__}: {exc}"
            print(f"form_without_entity_maps -> {no_map_err}")
        try:
            fem.form(integrand * dx(2),
                     entity_maps={sub: np.arange(por.size, dtype=np.int32)})
            print("form_with_dict_entity_maps_compiled=True")
        except TypeError as exc:
            dict_err = f"{type(exc).__name__}: {exc}"
            print("form_with_dict_entity_maps -> "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
            print("petsc_typeerror_overloads>>>")
            print(str(exc))
            print("<<<petsc_typeerror_overloads")
    else:
        print("mutation=compiling_with_entity_maps_sequence_from_the_start")

    f_sub = fem.form(integrand * dx(2), entity_maps=[cell_map])
    b_sub = fem.assemble_vector(f_sub)
    sub_val = float(b_sub.array.sum())

    f_all = fem.form(integrand * ufl.dx(domain=msh), entity_maps=[cell_map])
    b_all = fem.assemble_vector(f_all)
    all_val = float(b_all.array.sum())

    u_sub = ufl.TrialFunction(Vs)
    A = fem.assemble_matrix(
        fem.form(u_sub * v_par * dx(2), entity_maps=[cell_map]))
    A.scatter_reverse()
    shape = tuple(A.to_dense().shape)

    print(f"sequence_entity_maps_sub_region_integral={sub_val:.6f}")
    print(f"sequence_entity_maps_whole_domain_integral={all_val:.6f}")
    print(f"coupling_block_shape={shape}")

    seq_works = np.isfinite(sub_val) and abs(sub_val - 1.0) < 1e-12
    rect = shape[0] != shape[1] and shape[0] > shape[1] > 0
    silent = abs(all_val - sub_val) < 1e-14
    overloads_mention_sequence = (
        "entity_maps: collections.abc.Sequence[dolfinx.cpp.mesh.EntityMap]"
        in dict_err)
    print(f"form_without_entity_maps_raised_runtimeerror={bool(no_map_err)}")
    print(f"dict_entity_maps_raised_typeerror={bool(dict_err)}")
    print(f"typeerror_names_the_entitymap_sequence={overloads_mention_sequence}")
    print(f"sequence_entity_maps_compiles_and_assembles={bool(seq_works)}")
    print(f"rectangular_coupling_block_assembles={bool(rect)}")
    print(f"whole_domain_integral_silently_equals_sub_region={bool(silent)}")
    if (no_map_err and dict_err and overloads_mention_sequence and seq_works
            and rect and silent):
        print("VERDICT=entity_maps_must_be_a_sequence_of_entitymap")
        return 0
    print("VERDICT=entity_maps_was_not_required_or_dict_still_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
