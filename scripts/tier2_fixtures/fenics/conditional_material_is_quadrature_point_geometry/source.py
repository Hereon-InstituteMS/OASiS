"""Tier-2 for fenics magnetostatics#4: material data written with
`ufl.conditional` is evaluated at QUADRATURE POINTS, so the effective geometry
of the material interface silently depends on the quadrature degree of the
form. A cell-wise DG0 fem.Function built from `mesh.compute_midpoints` pins the
interface to cell boundaries instead.

The same coil-in-iron problem is solved twice on the same 16x16 mesh, once with
a ufl.conditional permeability and once with a DG0 permeability, at P1 and then
at P2. The two materials agree BIT-FOR-BIT at P1 (the P1 form has a single
midpoint quadrature point per cell, which is exactly what the DG0 marker uses)
and disagree at P2.

Mutation control: T2_MUTATE=1 uses the DG0 material for both solves, i.e. the
recommended fix; the P2 difference then vanishes too.
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

MU0 = 4.0e-7 * np.pi
J0 = 1.0e6
R_COIL = 0.2
NU_AIR = 1.0 / MU0
NU_IRON = 1.0 / (MU0 * 1000.0)
N = 16


def solve(deg: int, material: str) -> float:
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [N, N])
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    mid = dolfinx.mesh.compute_midpoints(
        msh, tdim, np.arange(ncells, dtype=np.int32)).T

    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    Jz = dolfinx.fem.Function(DG0)
    Jz.x.array[:] = 0.0
    Jz.x.array[(mid[0] ** 2 + mid[1] ** 2) < R_COIL ** 2] = J0

    if material == "dg0":
        nu_f = dolfinx.fem.Function(DG0)
        nu_f.x.array[:] = NU_AIR
        m = np.maximum(np.abs(mid[0]), np.abs(mid[1]))
        nu_f.x.array[(m > 0.25) & (m < 0.40)] = NU_IRON
        nu = nu_f
    elif material == "conditional":
        x = ufl.SpatialCoordinate(msh)
        m = ufl.max_value(abs(x[0]), abs(x[1]))
        nu = ufl.conditional(
            ufl.And(ufl.gt(m, 0.25), ufl.lt(m, 0.40)), NU_IRON, NU_AIR)
    else:
        raise AssertionError(material)

    V = dolfinx.fem.functionspace(msh, ("Lagrange", deg))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = nu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = Jz * v * ufl.dx
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), bdofs, V)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=f"t2_ms4_{deg}{material}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    Az = prob.solve()
    assert prob.solver.getConvergedReason() > 0
    return float(Az.x.array.max())


def main() -> int:
    tested = "dg0" if MUTATE else "conditional"
    print(f"material_under_test={tested}")
    out = {}
    for deg in (1, 2):
        a = solve(deg, tested)
        b = solve(deg, "dg0")
        rel = abs(a - b) / abs(b)
        out[deg] = rel
        print(f"degree={deg} under_test_max_Az={a:.10f} dg0_max_Az={b:.10f} "
              f"relative_difference={rel:.3e}")
    p1_identical = out[1] == 0.0
    p2_differs = out[2] > 1e-3
    print(f"p1_answers_are_bit_identical={p1_identical}")
    print(f"p2_relative_difference_exceeds_1e_3={p2_differs}")
    if p1_identical and p2_differs:
        print("VERDICT=conditional_material_geometry_follows_the_quadrature_"
              "degree")
        return 0
    print("VERDICT=material_spelling_does_not_change_the_answer")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
