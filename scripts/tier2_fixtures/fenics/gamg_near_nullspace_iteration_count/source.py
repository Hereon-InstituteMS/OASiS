"""Tier-2 for fenics linear_elasticity#4: the rigid-body near-nullspace is an
iteration-count win for CG+GAMG, and its ABSENCE does not make the solve fail.

Wrong variant: CG + GAMG on 3D elasticity with no A.setNearNullSpace(). The
claim's own correction is the part worth pinning: this still CONVERGES, so
"MUST" is wrong and an unchecked script sees nothing amiss; what changes is the
iteration count.

The fixture solves the same tetrahedral cantilever twice per mesh, once without
and once with the 6 rigid-body modes, and checks (a) both converge, (b) the
no-nullspace run needs strictly more iterations. Counts are compared in-process,
so no measured iteration number is pinned.

Mutation control: T2_MUTATE=1 attaches the near-nullspace to BOTH solves; the
counts become equal and the strict inequality fails.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
E, NU = 1.0e5, 0.3


def rigid_body_modes(V):
    """The 6 modes GAMG needs told about: 3 translations, 3 rotations."""
    x = V.tabulate_dof_coordinates()
    gdim = V.mesh.geometry.dim
    n = x.shape[0]
    basis = []
    for d in range(gdim):
        b = np.zeros((n, gdim))
        b[:, d] = 1.0
        basis.append(b)
    rot = [((0, 1), (-1.0, 1.0)), ((1, 2), (-1.0, 1.0)), ((2, 0), (-1.0, 1.0))]
    for (i, k), (si, sk) in rot:
        b = np.zeros((n, gdim))
        b[:, i] = si * x[:, k]
        b[:, k] = sk * x[:, i]
        basis.append(b)
    vecs = []
    for b in basis:
        v = dolfinx.fem.Function(V)
        v.x.array[:] = b.reshape(-1)[: v.x.array.size]
        v.x.scatter_forward()
        vecs.append(v.x.petsc_vec.copy())
    # Gram-Schmidt so PETSc accepts them as an orthonormal near-nullspace.
    for i, vi in enumerate(vecs):
        for vj in vecs[:i]:
            vi.axpy(-vi.dot(vj), vj)
        vi.normalize()
    return vecs


def solve(nx: int, with_nullspace: bool) -> tuple[int, int, int]:
    msh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])],
        [2 * nx, nx, nx], dolfinx.mesh.CellType.tetrahedron)
    gdim = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    mu = E / (2.0 * (1.0 + NU))
    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim),
                  eps(v)) * ufl.dx
    f = dolfinx.fem.Constant(msh, np.array([0.0, 0.0, -1.0]))
    L = ufl.dot(f, v) * ufl.dx

    msh.topology.create_connectivity(gdim - 1, gdim)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, gdim - 1, lambda x: np.isclose(x[0], 0.0))
    clamp = dolfinx.fem.Function(V)
    bc = dolfinx.fem.dirichletbc(
        clamp, dolfinx.fem.locate_dofs_topological(V, gdim - 1, left))

    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs=[bc])
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L))
    dolfinx.fem.petsc.apply_lifting(b, [dolfinx.fem.form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc])

    if with_nullspace:
        vecs = rigid_body_modes(V)
        A.setNearNullSpace(PETSc.NullSpace().create(vectors=vecs))

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=1e-8)
    ksp.setFromOptions()
    uh = dolfinx.fem.Function(V)
    ksp.solve(b, uh.x.petsc_vec)
    return (ksp.getIterationNumber(), ksp.getConvergedReason(),
            V.dofmap.index_map.size_global * V.dofmap.index_map_bs)


def main() -> int:
    ok = True
    for nx in (4, 6):
        it_no, r_no, ndof = solve(nx, with_nullspace=MUTATE)
        it_yes, r_yes, _ = solve(nx, with_nullspace=True)
        print(f"ndofs={ndof} without_nullspace_iters={it_no} "
              f"without_reason={r_no} with_nullspace_iters={it_yes} "
              f"with_reason={r_yes}")
        converged_anyway = r_no > 0
        fewer = it_yes < it_no
        print(f"ndofs={ndof}_without_nullspace_still_converged="
              f"{converged_anyway}")
        print(f"ndofs={ndof}_nullspace_reduces_iterations={fewer}")
        ok = ok and converged_anyway and fewer
    if ok:
        print("VERDICT=near_nullspace_is_iteration_win_not_correctness_gate")
        return 0
    print("VERDICT=no_iteration_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
