"""Tier-2 for fenics maxwell#4: N1curl degrees of freedom are EDGE MOMENTS, not
nodal components, so a tangential boundary condition has to be interpolated into
the edge basis. Writing component values into the dof array instead leaves the
tangential trace wrong with no complaint from anything.

FINDING against the claim text: the claim says a dirichletbc "defined with a
vector-valued function silently sets only the first component on each edge".
That does NOT reproduce on dolfinx 0.10.0 -- the vector-CONSTANT spelling is
rejected outright with RuntimeError: Creating a DirichletBC using a Constant is
not supported when the Constant size is not equal to the block size of the
constrained (sub-)space. Use a fem::Function to create the fem::DirichletBC.
The silent version of the mistake is one level lower: filling the boundary dofs
with the intended component value by hand.

Wrong variant: f.x.array[boundary_dofs] = 1.0 for the intended uniform field
E = (1, 0). Observed: the boundary tangential trace error is O(1) (the correct
edge moments for that field are 0 and +-0.125 on an 8x8 mesh, never 1.0), and
nothing raises.

Mutation control: T2_MUTATE=1 sets the same boundary values by interpolating
(1, 0) into the N1curl space, which reproduces the tangential trace exactly and
removes the error signal.
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

N = 8


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(
        msh, basix.ufl.element("N1curl", msh.basix_cell(), 1))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    print(f"n_boundary_dofs={len(bdofs)} n_dofs={V.dofmap.index_map.size_global}")

    # the vector-Constant spelling is refused, so it cannot be the silent bug
    const_msg = ""
    try:
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, (1.0, 0.0)), bdofs, V)
        print("constant_spelling_raised=False")
    except Exception as exc:
        const_msg = str(exc).splitlines()[0]
        print(f"constant_spelling_raised=True {type(exc).__name__}: "
              f"{const_msg}")

    # what the correct edge moments of the uniform field (1, 0) actually are
    g = dolfinx.fem.Function(V)
    g.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
                                       np.zeros_like(x[0]))))
    edge_moments = g.x.array[bdofs]
    print(f"edge_moment_values_min_max="
          f"{float(edge_moments.min()):.5f},{float(edge_moments.max()):.5f}")
    moments_not_components = (float(np.abs(edge_moments).max()) < 0.5
                              and float(edge_moments.min()) < 0.0)
    print(f"dof_values_are_edge_moments_not_components="
          f"{moments_not_components}")

    n = ufl.FacetNormal(msh)
    tang = ufl.as_vector((-n[1], n[0]))
    exact = ufl.as_vector((1.0, 0.0))

    def trace_error(fun) -> float:
        frm = dolfinx.fem.form(ufl.inner(fun - exact, tang) ** 2 * ufl.ds)
        return float(np.sqrt(abs(dolfinx.fem.assemble_scalar(frm))))

    raised = ""
    f = dolfinx.fem.Function(V)
    f.x.array[:] = 0.0
    try:
        if MUTATE:
            bc = dolfinx.fem.dirichletbc(g, bdofs)
            dolfinx.fem.set_bc(f.x.array, [bc])
            print("boundary_values_set_by=interpolation_into_n1curl")
        else:
            f.x.array[bdofs] = 1.0
            print("boundary_values_set_by=handwritten_component_value")
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
        print(f"setting_boundary_values_raised=True {raised}")
    print(f"setting_boundary_values_raised={bool(raised)}")

    err_test = trace_error(f)
    ok = dolfinx.fem.Function(V)
    ok.x.array[:] = 0.0
    dolfinx.fem.set_bc(ok.x.array, [dolfinx.fem.dirichletbc(g, bdofs)])
    err_ref = trace_error(ok)
    print(f"tangential_trace_error_under_test={err_test:.6f}")
    print(f"tangential_trace_error_interpolated={err_ref:.6e}")
    print(f"interpolated_bc_reproduces_the_tangential_trace={err_ref < 1e-10}")
    print(f"under_test_tangential_trace_is_wrong={err_test > 0.5}")

    if (const_msg and moments_not_components and not raised
            and err_ref < 1e-10 and err_test > 0.5):
        print("VERDICT=edge_moments_must_be_interpolated_handwritten_dofs_are_"
              "silently_wrong")
        return 0
    print("VERDICT=handwritten_dof_values_were_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
