"""Tier-2 for fenics eigenvalue#1: a shift-and-invert spectral transformation is
what makes SLEPc look at INTERIOR eigenvalues. Setting eps.setTarget(sigma) on
the dolfinx-assembled stiffness matrix and solving with the default spectral
transform returns the extreme eigenvalues instead.

Wrong variant: eps.setTarget(sigma) alone, default ST (plain shift) and default
which (LARGEST_MAGNITUDE). Right variant: eps.setTarget(sigma) plus
eps.setWhichEigenpairs(TARGET_MAGNITUDE) plus
eps.getST().setType(SLEPc.ST.Type.SINVERT).

16x16 P1 Dirichlet Laplacian stiffness matrix (289 dofs, assembled with
bcs=[bc], diag left at 1). sigma = 3.0 sits inside the spectrum; the nearest
true eigenvalues are 2.9760 and 2.9176.

Observed on dolfinx 0.10.0 / slepc4py 3.24.3: setTarget alone converges in 7
iterations to 7.9231, 7.8093, 7.6955, ... which is exactly the top of the
spectrum, 4.92 away from the target. With TARGET_MAGNITUDE and SINVERT the same
solve returns 2.9760, 2.9760, 2.9176, ... in 4 iterations. Note two things the
claim does not say: SINVERT here must use a pivoting factoriser (the shifted
matrix is indefinite, and PETSc's own LU stops with "Zero pivot row 8 value 0"),
and if you set TARGET_MAGNITUDE but skip SINVERT the solve still finds the
target cluster — it just needs 57 iterations instead of 4.

Mutation control: T2_MUTATE=1 gives the primary solve the full correct recipe,
so it no longer returns the extreme of the spectrum and the fixture loses its
own expectation.
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

from petsc4py import PETSc  # noqa: E402
from slepc4py import SLEPc  # noqa: E402

N = 16
TARGET = 3.0
NEV = 5


def stiffness():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 0.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, bfacets), V)
    A = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx),
        bcs=[bc])
    A.assemble()
    return msh, A


def eig(A, msh, target=None, which=None, sinvert=False, nev=NEV):
    eps = SLEPc.EPS().create(msh.comm)
    eps.setOperators(A)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setDimensions(nev, PETSc.DECIDE)
    eps.setTolerances(1.0e-8, 1000)
    if target is not None:
        eps.setTarget(target)
    if which is not None:
        eps.setWhichEigenpairs(which)
    if sinvert:
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")   # shifted matrix is indefinite
    eps.solve()
    n = eps.getConverged()
    vals = np.array([eps.getEigenvalue(i).real for i in range(n)])
    return eps.getConvergedReason(), n, vals, eps.getIterationNumber()


def main() -> int:
    msh, A = stiffness()

    # Reference: where the extremes of this matrix actually are.
    _, _, big, _ = eig(A, msh, which=SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    spectrum_max = float(np.max(big))

    if MUTATE:
        r_p, n_p, v_p, it_p = eig(A, msh, target=TARGET,
                                  which=SLEPc.EPS.Which.TARGET_MAGNITUDE,
                                  sinvert=True)
    else:
        r_p, n_p, v_p, it_p = eig(A, msh, target=TARGET)
    r_s, n_s, v_s, it_s = eig(A, msh, target=TARGET,
                              which=SLEPc.EPS.Which.TARGET_MAGNITUDE,
                              sinvert=True)
    r_n, n_n, v_n, it_n = eig(A, msh, target=TARGET,
                              which=SLEPc.EPS.Which.TARGET_MAGNITUDE,
                              sinvert=False)

    print(f"target={TARGET} spectrum_largest={spectrum_max:.4f}")
    print(f"primary reason={r_p} nconv={n_p} its={it_p} "
          f"first_values={np.round(v_p[:3], 4).tolist()}")
    print(f"sinvert reason={r_s} nconv={n_s} its={it_s} "
          f"first_values={np.round(v_s[:3], 4).tolist()}")
    print(f"target_magnitude_without_sinvert its={it_n} "
          f"first_values={np.round(v_n[:3], 4).tolist()}")

    d_p = float(np.max(np.abs(v_p - TARGET))) if n_p else float("nan")
    d_s = float(np.max(np.abs(v_s - TARGET))) if n_s else float("nan")
    print(f"primary_max_distance_from_target={d_p:.4f} "
          f"sinvert_max_distance_from_target={d_s:.4f}")

    hit_extreme = n_p > 0 and abs(float(np.max(v_p)) - spectrum_max) < 1.0e-8
    far = d_p > 1.0
    clustered = n_s >= NEV and d_s < 0.1 * TARGET
    cheaper = it_n >= 5 * max(it_s, 1)
    print(f"primary_returned_the_largest_eigenvalue_of_the_matrix={hit_extreme}")
    print(f"primary_stayed_more_than_one_away_from_the_target={far}")
    print(f"sinvert_clustered_inside_ten_percent_of_the_target={clustered}")
    print(f"sinvert_needed_at_least_five_times_fewer_iterations={cheaper}")
    if hit_extreme and far and clustered and cheaper:
        print("VERDICT=sinvert_is_what_moves_slepc_onto_the_interior_target")
        return 0
    print("VERDICT=target_alone_already_found_the_interior_eigenvalues")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
