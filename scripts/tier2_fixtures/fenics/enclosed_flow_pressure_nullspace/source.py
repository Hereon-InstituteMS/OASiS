"""Tier-2 for fenics navier_stokes#5: with velocity Dirichlet data on every
boundary facet the pressure is determined only up to a constant, because
pressure enters the weak form only through its gradient.

Rather than depend on how a particular solver reacts to a singular saddle-point
matrix, the fixture measures the null space directly: build the constant-
pressure vector (1 on every pressure dof, 0 on velocity dofs) and apply the
assembled operator to it. If the constant really is in the kernel, the result is
at round-off relative to the matrix norm.

Mutation control: T2_MUTATE=1 pins one pressure dof, which removes the kernel;
the same product is then O(1).
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



def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)

    V0, _ = W.sub(0).collapse()
    lid = dolfinx.fem.Function(V0)
    lid.interpolate(lambda x: np.vstack(
        [np.isclose(x[1], 1.0) * 1.0, np.zeros_like(x[0])]))
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(
        (W.sub(0), V0), tdim - 1, facets)
    bcs = [dolfinx.fem.dirichletbc(lid, dofs, W.sub(0))]

    Q0, q_map = W.sub(1).collapse()
    if MUTATE:
        pin = dolfinx.fem.Function(Q0)
        bcs.append(dolfinx.fem.dirichletbc(
            pin, np.array([q_map[0]], dtype=np.int32), W.sub(1)))

    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs=bcs)
    A.assemble()
    z = dolfinx.fem.Function(W)
    z.x.array[:] = 0.0
    z.x.array[np.array(q_map, dtype=np.int32)] = 1.0
    out = A.createVecLeft()
    A.mult(z.x.petsc_vec, out)
    rel = out.norm() / A.norm()
    print(f"pressure_pinned={MUTATE}")
    print(f"n_pressure_dofs={len(q_map)}")
    print(f"relative_norm_of_A_times_constant_pressure={rel:.3e}")
    in_kernel = rel < 1e-12
    print(f"constant_pressure_is_in_the_kernel={in_kernel}")
    if in_kernel:
        print("VERDICT=enclosed_flow_leaves_the_pressure_constant_free")
        return 0
    print("VERDICT=pressure_is_determined")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
