"""Tier-2 for fenics navier_stokes#7: a Dirichlet condition on a sub-space of a
mixed (Taylor-Hood) space needs BOTH a Function on the COLLAPSED sub-space and
the tuple form of locate_dofs_*.

Three wrong spellings are tried on one Taylor-Hood space -- a raw numpy array, a
fem.Constant, and a Function on the collapsed space combined with a single
(non-tuple) dof array. Observed signal: all three raise the same TypeError,
beginning "__init__(): incompatible function arguments. The following argument
types are supported:", listing four overloads, and ending with a line beginning
"Invoked with types: dolfinx.cpp.fem.DirichletBC_float64,".

The trap the fixture also pins: a Function built on the UNCOLLAPSED mixed space
W does NOT raise, so the absence of a TypeError is not proof that the BC is
right. Each of the two accepted spellings is applied with DirichletBC.set to a
zeroed Function on the Taylor-Hood space and the imposed velocity dofs are read
back: the correct one writes the lid speed, the silently-accepted uncollapsed-W
one writes zero.

Mutation control: T2_MUTATE=1 uses the correct spelling everywhere, so nothing
raises and the boundary velocity is the intended one.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
LID = 2.0


def lid_values(x):
    return np.vstack([np.isclose(x[1], 1.0) * LID, np.zeros_like(x[0])])


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    gdim, tdim = msh.geometry.dim, msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
    V0, v_map = W.sub(0).collapse()
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs_pair = dolfinx.fem.locate_dofs_topological(
        (W.sub(0), V0), tdim - 1, facets)
    dofs_single = dolfinx.fem.locate_dofs_topological(
        W.sub(0), tdim - 1, facets)
    u_bc = dolfinx.fem.Function(V0)
    u_bc.interpolate(lid_values)

    correct = [("collapsed_Function_and_dof_pair", u_bc, dofs_pair)]
    wrong = [
        ("raw_numpy_array", np.zeros(gdim, dtype=np.float64), dofs_pair),
        ("fem_Constant", dolfinx.fem.Constant(msh, (0.0,) * gdim), dofs_pair),
        ("collapsed_Function_single_dof_array", u_bc, dofs_single),
    ]
    trials = correct if MUTATE else wrong

    messages = []
    n_raised = 0
    for name, value, dofs in trials:
        try:
            dolfinx.fem.dirichletbc(value, dofs, W.sub(0))
            print(f"spelling={name} raised=False")
        except Exception as exc:  # noqa: BLE001
            n_raised += 1
            messages.append(str(exc))
            first = str(exc).splitlines()[0]
            invoked = [ln for ln in str(exc).splitlines()
                       if ln.startswith("Invoked with types:")]
            print(f"spelling={name} raised=True {type(exc).__name__}: {first}")
            print(f"  {invoked[0] if invoked else '(no Invoked line)'}")

    heads = {m.split("Invoked with types:")[0] for m in messages}
    invoked_ok = all(
        "Invoked with types: dolfinx.cpp.fem.DirichletBC_float64," in m
        for m in messages)
    all_same = bool(messages) and len(heads) == 1
    print(f"wrong_spellings_raised={n_raised}")
    print(f"all_three_raise_the_same_overload_list={all_same}")
    print(f"all_three_invoked_line_names_DirichletBC_float64={invoked_ok}")

    # The silent trap: a Function on the uncollapsed mixed space is accepted.
    u_wrong = dolfinx.fem.Function(W)
    u_wrong.x.array[:] = 0.0
    if MUTATE:
        bc_quiet = dolfinx.fem.dirichletbc(u_bc, dofs_pair, W.sub(0))
        quiet = True
    else:
        try:
            bc_quiet = dolfinx.fem.dirichletbc(u_wrong, dofs_pair, W.sub(0))
            quiet = True
        except Exception as exc:  # noqa: BLE001
            bc_quiet, quiet = None, False
            print(f"uncollapsed_W_Function_raised=True {exc}")
    print(f"uncollapsed_W_Function_is_accepted_without_error={quiet}")

    # What does the accepted BC actually write into the solution vector?
    w_quiet = dolfinx.fem.Function(W)
    w_quiet.x.array[:] = 0.0
    bc_quiet.set(w_quiet.x.array)
    w_ref = dolfinx.fem.Function(W)
    w_ref.x.array[:] = 0.0
    dolfinx.fem.dirichletbc(u_bc, dofs_pair, W.sub(0)).set(w_ref.x.array)
    vdofs = np.array(v_map, dtype=np.int32)
    speed = float(np.max(np.abs(w_quiet.x.array[vdofs])))
    speed_ref = float(np.max(np.abs(w_ref.x.array[vdofs])))
    print(f"max_velocity_dof_from_accepted_bc={speed:.6f} "
          f"from_correct_bc={speed_ref:.6f} imposed_lid={LID}")
    lid_lost = speed < 0.5 * LID and abs(speed_ref - LID) < 1e-12
    print(f"accepted_bc_lost_the_lid_velocity={lid_lost}")

    if n_raised == 3 and all_same and invoked_ok and quiet and lid_lost:
        print("VERDICT=subspace_bc_needs_collapsed_function_and_dof_pair")
        return 0
    print("VERDICT=subspace_bc_spellings_agree")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
