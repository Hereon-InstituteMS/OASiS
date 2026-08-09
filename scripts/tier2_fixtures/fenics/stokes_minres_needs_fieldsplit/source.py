"""Tier-2 for fenics stokes#3: a block preconditioner is what makes an iterative
Stokes solve scale. Without PCFIELDSPLIT the saddle-point spectrum makes the
MinRes iteration count grow with every refinement; with a fieldsplit that puts
an AMG solve on the velocity block and the pressure mass matrix on the pressure
block, the count stops moving.

Wrong variant: MinRes with a point Jacobi preconditioner on the monolithic
Taylor-Hood matrix. Right variant: PCFIELDSPLIT over the velocity/pressure
index sets, hypre BoomerAMG on the velocity block and the pressure mass matrix
M_p on the other, both as an additive block-diagonal MinRes preconditioner and
as a Schur-complement FGMRES preconditioner with
PC.SchurPreType.USER = M_p.

P2/P1 channel flow on the unit square, parabolic inflow at x = 0, no-slip on
y = 0 and y = 1, natural outflow at x = 1, rtol 1e-8, N = 8 / 16 / 32.

Observed on dolfinx 0.10.0 / PETSc 3.24.5: Jacobi-MinRes takes 511, 1097 and
2094 iterations at 659, 2467 and 9539 dofs — it roughly doubles per halving of
h. The additive fieldsplit takes 59, 65, 67 and the Schur fieldsplit takes 47,
51, 49. One correction to the claim: that growth is O(h^-1), not the O(h^-2) the
claim states; the mechanism holds, the exponent does not. Note also that
PCFIELDSPLIT of type Schur needs SORTED index sets — the map returned by
W.sub(0).collapse() is not sorted, and passing it straight through makes
PCSetUp_FieldSplit stop in ISComplement with "Index set must be sorted".

Mutation control: T2_MUTATE=1 makes the additive fieldsplit the primary
preconditioner, its iteration count stops growing and the fixture loses its own
expectation.
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

import basix.ufl  # noqa: E402
from petsc4py import PETSc  # noqa: E402

MESHES = (8, 16, 32)
RTOL = 1.0e-8
MAXIT = 4000


def build(n):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    gdim = msh.geometry.dim
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v))
         - p * ufl.div(v) - q * ufl.div(u)) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(msh, np.zeros(gdim)), v) * ufl.dx

    V, vmap = W.sub(0).collapse()
    Q, qmap = W.sub(1).collapse()
    walls = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    inlet = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    zero = dolfinx.fem.Function(V)
    zero.x.array[:] = 0.0
    parabolic = dolfinx.fem.Function(V)
    parabolic.interpolate(lambda x: np.vstack(
        [4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
    bcs = [
        dolfinx.fem.dirichletbc(
            zero,
            dolfinx.fem.locate_dofs_topological((W.sub(0), V), fdim, walls),
            W.sub(0)),
        dolfinx.fem.dirichletbc(
            parabolic,
            dolfinx.fem.locate_dofs_topological((W.sub(0), V), fdim, inlet),
            W.sub(0)),
    ]
    af, Lf = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    pp, qq = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    Mp = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(pp * qq * ufl.dx))
    Mp.assemble()
    # PCFIELDSPLIT(Schur) calls ISComplement, which requires sorted index sets.
    isu = PETSc.IS().createGeneral(
        np.sort(np.asarray(vmap, dtype=PETSc.IntType)), comm=msh.comm)
    isp = PETSc.IS().createGeneral(
        np.sort(np.asarray(qmap, dtype=PETSc.IntType)), comm=msh.comm)
    ndof = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
    return msh, W, A, b, Mp, isu, isp, ndof


def solve(msh, W, A, b, Mp, isu, isp, kind):
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setTolerances(rtol=RTOL, max_it=MAXIT)
    pc = ksp.getPC()
    if kind == "minres_jacobi":
        ksp.setType("minres")
        pc.setType("jacobi")
    elif kind == "minres_fieldsplit":
        ksp.setType("minres")
        pc.setType("fieldsplit")
        pc.setFieldSplitIS(("u", isu), ("p", isp))
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
        pc.setUp()
        sub = pc.getFieldSplitSubKSP()
        sub[0].setType("preonly")
        sub[0].getPC().setType("hypre")
        sub[1].setType("preonly")
        sub[1].getPC().setType("jacobi")
        sub[1].setOperators(Mp, Mp)
    elif kind == "fgmres_schur":
        ksp.setType("fgmres")
        pc.setType("fieldsplit")
        pc.setFieldSplitIS(("u", isu), ("p", isp))
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.UPPER)
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, Mp)
        pc.setUp()
        sub = pc.getFieldSplitSubKSP()
        sub[0].setType("preonly")
        sub[0].getPC().setType("hypre")
        sub[1].setType("preonly")
        sub[1].getPC().setType("jacobi")
    w = dolfinx.fem.Function(W)
    ksp.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()
    return int(ksp.getConvergedReason()), int(ksp.getIterationNumber())


def main() -> int:
    primary_kind = "minres_fieldsplit" if MUTATE else "minres_jacobi"
    prim, add, schur, dofs = [], [], [], []
    for n in MESHES:
        msh, W, A, b, Mp, isu, isp, ndof = build(n)
        r_p, it_p = solve(msh, W, A, b, Mp, isu, isp, primary_kind)
        r_a, it_a = solve(msh, W, A, b, Mp, isu, isp, "minres_fieldsplit")
        r_s, it_s = solve(msh, W, A, b, Mp, isu, isp, "fgmres_schur")
        prim.append(it_p)
        add.append(it_a)
        schur.append(it_s)
        dofs.append(ndof)
        print(f"N={n:3d} ndofs={ndof:6d} primary(reason={r_p}) its={it_p} "
              f"additive_fieldsplit(reason={r_a}) its={it_a} "
              f"schur_fieldsplit(reason={r_s}) its={it_s}")

    growth = [prim[i + 1] / prim[i] for i in range(len(prim) - 1)]
    print("primary_iteration_growth_per_refinement="
          + " ".join(f"{g:.2f}" for g in growth))
    print(f"additive_fieldsplit_iterations={add} "
          f"schur_fieldsplit_iterations={schur}")

    grows = all(g > 1.6 for g in growth)
    add_flat = max(add) <= 1.3 * min(add)
    schur_flat = max(schur) <= 1.3 * min(schur)
    much_cheaper = prim[-1] > 10 * schur[-1]
    print(f"primary_iterations_grow_by_more_than_1p6x_per_refinement={grows}")
    print(f"additive_fieldsplit_iterations_stay_within_30_percent={add_flat}")
    print(f"schur_fieldsplit_iterations_stay_within_30_percent={schur_flat}")
    print(f"primary_needs_more_than_ten_times_the_schur_iterations="
          f"{much_cheaper}")
    if grows and add_flat and schur_flat and much_cheaper:
        print("VERDICT=minres_without_fieldsplit_scales_with_the_mesh")
        return 0
    print("VERDICT=minres_without_fieldsplit_already_scaled")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
