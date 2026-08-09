"""Tier-2 for fenics mixed_poisson#3: the mixed Poisson system is INDEFINITE.
Conjugate gradients is only valid for a symmetric POSITIVE DEFINITE operator,
so PETSc's CG bails out on it, and a plain preconditioned GMRES cannot reach a
tight tolerance either — while the same matrix factorises fine under LU.

Wrong variant: ksp_type=cg with a plain preconditioner. Right variant, and the
mutation: a direct LU factorisation (here MUMPS).

Well-posed RT1 x DG0 on a 16x16 unit square, flux prescribed on x = 0 and
x = 1, natural pressure condition on y = 0 and y = 1, rtol 1e-8, at most 500
iterations.

Observed on dolfinx 0.10.0 / PETSc 3.24.5: CG with ILU stops after 2 iterations
with KSPConvergedReason -8, which is KSP_DIVERGED_INDEFINITE_PC, and CG with
Jacobi stops after 2 with -10, KSP_DIVERGED_INDEFINITE_MAT — both of them the
"this operator is not positive definite" exits. LU converges in a single
application and gives a sane pressure range of [-8.9315e-01, 9.2133e-01]. One
correction to the claim: GMRES does not literally stagnate. With the default
restart of 30 it exhausts all 500 iterations, exits with KSPConvergedReason -3
(DIVERGED_ITS) and has only pulled the residual from 1.228e+01 to 1.675e-05,
short of the 1e-8 relative tolerance; widen the restart to 200 and it does
converge, in 326 iterations against LU's one application. So the right way to
say it is that GMRES on this operator is expensive and restart-sensitive, not
that its residual fails to drop.

Mutation control: T2_MUTATE=1 makes the direct LU solve the primary one, so no
indefinite exit is reported and the fixture loses its own expectation.
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

N = 16
DEGREE = 1
RTOL = 1.0e-8
MAXIT = 500


def build():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    RT = basix.ufl.element("RT", msh.basix_cell(), DEGREE)
    DG = basix.ufl.element("DG", msh.basix_cell(), DEGREE - 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([RT, DG]))
    (sig, u) = ufl.TrialFunctions(W)
    (tau, v) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(msh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    nrm = ufl.FacetNormal(msh)
    a = (ufl.inner(sig, tau) + ufl.div(tau) * u + ufl.div(sig) * v) * ufl.dx
    natural = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.isclose(X[1], 0.0) | np.isclose(X[1], 1.0))
    flux = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0))
    tags = dolfinx.mesh.meshtags(msh, fdim, np.sort(natural),
                                 np.full(len(natural), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    L = -f * v * ufl.dx - ufl.sin(5.0 * x[0]) * ufl.dot(tau, nrm) * ds(1)
    V0, _ = W.sub(0).collapse()
    g = dolfinx.fem.Function(V0)
    g.x.array[:] = 0.0
    bcs = [dolfinx.fem.dirichletbc(
        g, dolfinx.fem.locate_dofs_topological((W.sub(0), V0), fdim, flux),
        W.sub(0))]
    af, Lf = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    return msh, W, A, b


def solve(msh, W, A, b, kind):
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setTolerances(rtol=RTOL, max_it=MAXIT)
    pc = ksp.getPC()
    if kind == "lu":
        ksp.setType("preonly")
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
    elif kind == "cg_ilu":
        ksp.setType("cg")
        pc.setType("ilu")
    elif kind == "cg_jacobi":
        ksp.setType("cg")
        pc.setType("jacobi")
    elif kind == "gmres_default_restart":
        ksp.setType("gmres")
        pc.setType("jacobi")
    elif kind == "gmres_restart_200":
        ksp.setType("gmres")
        ksp.setGMRESRestart(200)
        pc.setType("jacobi")
    ksp.setConvergenceHistory()
    w = dolfinx.fem.Function(W)
    ksp.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()
    hist = ksp.getConvergenceHistory()
    p = np.array(w.sub(1).collapse().x.array)
    return (int(ksp.getConvergedReason()), int(ksp.getIterationNumber()),
            (float(hist[0]), float(hist[-1])) if len(hist) else (0.0, 0.0),
            (float(np.min(p)), float(np.max(p))))


def main() -> int:
    msh, W, A, b = build()
    primary_kind = "lu" if MUTATE else "cg_ilu"
    r_p, it_p, h_p, rng_p = solve(msh, W, A, b, primary_kind)
    r_j, it_j, h_j, _ = solve(msh, W, A, b, "cg_jacobi")
    r_g, it_g, h_g, _ = solve(msh, W, A, b, "gmres_default_restart")
    r_g2, it_g2, h_g2, _ = solve(msh, W, A, b, "gmres_restart_200")
    r_l, it_l, _, rng_l = solve(msh, W, A, b, "lu")
    print(f"primary({primary_kind}) reason={r_p} its={it_p} "
          f"residual_first={h_p[0]:.3e} residual_last={h_p[1]:.3e}")
    print(f"cg_jacobi reason={r_j} its={it_j} "
          f"residual_first={h_j[0]:.3e} residual_last={h_j[1]:.3e}")
    print(f"gmres_default_restart reason={r_g} its={it_g} "
          f"residual_first={h_g[0]:.3e} residual_last={h_g[1]:.3e}")
    print(f"gmres_restart_200 reason={r_g2} its={it_g2} "
          f"residual_first={h_g2[0]:.3e} residual_last={h_g2[1]:.3e}")
    print(f"lu reason={r_l} its={it_l} "
          f"pressure_range=[{rng_l[0]:.4e}, {rng_l[1]:.4e}]")

    indef_pc = int(PETSc.KSP.ConvergedReason.DIVERGED_INDEFINITE_PC)
    indef_mat = int(PETSc.KSP.ConvergedReason.DIVERGED_INDEFINITE_MAT)
    primary_indefinite = r_p in (indef_pc, indef_mat)
    jacobi_indefinite = r_j in (indef_pc, indef_mat)
    gmres_short = r_g < 0 and it_g >= MAXIT and h_g[1] > RTOL * h_g[0]
    gmres_slow = it_g2 > 100
    lu_fine = r_l > 0 and abs(rng_l[1]) < 1.0e3 and abs(rng_l[0]) < 1.0e3
    print(f"primary_exited_with_an_indefinite_reason_code={primary_indefinite}")
    print(f"cg_with_jacobi_also_exited_indefinite={jacobi_indefinite}")
    print(f"gmres_with_the_default_restart_never_reached_the_tolerance="
          f"{gmres_short}")
    print(f"gmres_needed_more_than_a_hundred_iterations_even_with_restart_200="
          f"{gmres_slow}")
    print(f"lu_factorised_the_same_matrix_and_gave_a_sane_pressure={lu_fine}")
    if (primary_indefinite and jacobi_indefinite and gmres_short and gmres_slow
            and lu_fine):
        print("VERDICT=mixed_poisson_saddle_point_is_indefinite_and_cg_refuses_it")
        return 0
    print("VERDICT=cg_handled_the_saddle_point")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
