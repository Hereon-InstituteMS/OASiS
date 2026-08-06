"""Tier-2 for fenics helmholtz#2: at high wavenumber the off-the-shelf
preconditioners give up on the Helmholtz operator, and the shifted-Laplacian
preconditioner is what makes GMRES work.

Wrong variant: GMRES with PETSc's default-ish ILU on k = 100, 32x32 unit square,
P1, complex build, Gaussian source, rtol 1e-8, 1000 iterations of un-restarted
GMRES. Observed on dolfinx 0.10.0 / complex PETSc: reason -3
(DIVERGED_MAX_IT) with a relative residual of 8.7e+00 -- the residual is nearly
an order of magnitude LARGER than the right-hand side after 1000 iterations.

The fix, measured in the same run: a shifted-Laplacian preconditioner, i.e. the
exact inverse of the SAME form assembled with the complex shift
(1 + 0.5i) k^2 instead of k^2, applied through a petsc4py python PC. It converges
in 197 iterations, where plain Jacobi needs 509 and ILU never gets there.

FINDING against the claim text: the claim promises the shifted Laplacian
"restores ~10 iterations per convergence". That is not what happens -- 197
iterations at k = 100, and the count scales with k (82 iterations at k = 50 on
the same mesh), which is the known O(k) behaviour of a beta = O(1) shift. The
claim's "stagnates at residual ~1e-2" is also only true of ILU here; Jacobi does
converge on a problem this small (1089 dofs), it just takes 2.6x the iterations
of the shifted solve.

Mutation control: T2_MUTATE=1 puts the shifted-Laplacian PC under test in place
of ILU; the divergence signal disappears.
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

N = 32
BETA = 0.5
MAX_IT = 1000


class ShiftedLaplacianPC:
    """Applies (Laplacian - (1 + i*beta) k^2 mass)^-1 by a direct solve."""

    def __init__(self, mat):
        self.ksp = PETSc.KSP().create(mat.getComm())
        self.ksp.setOperators(mat)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType("mumps")
        self.ksp.setUp()

    def apply(self, pc, x, y):  # noqa: D401 - petsc4py interface
        self.ksp.solve(x, y)


def operator(k: float, coefficient):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v))
         - coefficient * ufl.inner(u, v)) * ufl.dx
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    A.assemble()
    f = dolfinx.fem.Function(V)
    f.interpolate(lambda x: np.exp(-200.0 * ((x[0] - 0.5) ** 2
                                             + (x[1] - 0.5) ** 2)))
    b = dolfinx.fem.petsc.assemble_vector(
        dolfinx.fem.form(ufl.inner(f, v) * ufl.dx))
    b.assemble()
    # V is returned so the mesh (and its MPI communicator) outlives the matrix
    return A, b, V


def run(k: float, pc_kind: str) -> tuple[int, int, float]:
    A, b, keep_v = operator(k, k ** 2)
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setGMRESRestart(MAX_IT)
    ksp.setTolerances(rtol=1e-8, max_it=MAX_IT)
    ctx = None
    if pc_kind == "shifted":
        S, _, keep_s = operator(k, (1.0 + BETA * 1j) * k ** 2)
        ctx = ShiftedLaplacianPC(S)
        pc = ksp.getPC()
        pc.setType("python")
        pc.setPythonContext(ctx)
        pc.setUp()
    else:
        ksp.getPC().setType(pc_kind)
    x = A.createVecRight()
    ksp.solve(b, x)
    rel = float(ksp.getResidualNorm() / b.norm())
    return ksp.getIterationNumber(), ksp.getConvergedReason(), rel


def main() -> int:
    print(f"scalar_type={np.dtype(dolfinx.default_scalar_type).name}")
    if not np.issubdtype(dolfinx.default_scalar_type, np.complexfloating):
        print("VERDICT=needs_the_complex_build")
        return 1

    kind = "shifted" if MUTATE else "ilu"
    print(f"preconditioner_under_test={kind} k=100 mesh={N}x{N} "
          f"max_iterations={MAX_IT}")
    it_u, r_u, rel_u = run(100.0, kind)
    print(f"under_test iterations={it_u} converged_reason={r_u} "
          f"relative_residual={rel_u:.3e}")
    print(f"under_test_pc_fails_to_converge={r_u < 0}")
    print(f"under_test_residual_exceeds_the_rhs={rel_u > 1.0}")

    it_j, r_j, rel_j = run(100.0, "jacobi")
    it_s, r_s, rel_s = run(100.0, "shifted")
    it_s50, r_s50, rel_s50 = run(50.0, "shifted")
    print(f"jacobi k=100 iterations={it_j} reason={r_j} "
          f"relative_residual={rel_j:.3e}")
    print(f"shifted_laplacian k=100 iterations={it_s} reason={r_s} "
          f"relative_residual={rel_s:.3e}")
    print(f"shifted_laplacian k=50 iterations={it_s50} reason={r_s50} "
          f"relative_residual={rel_s50:.3e}")

    print(f"shifted_laplacian_converges={r_s > 0}")
    print(f"shifted_laplacian_beats_jacobi={it_s < it_j}")
    print(f"shifted_laplacian_converged_in_about_ten_iterations={it_s <= 20}")
    print(f"shifted_laplacian_iteration_count_grows_with_k="
          f"{it_s > 1.5 * it_s50}")

    if (r_u < 0 and rel_u > 1.0 and r_s > 0 and it_s < it_j
            and it_s > 1.5 * it_s50):
        print("VERDICT=default_pc_diverges_and_the_shifted_laplacian_carries_"
              "the_solve")
        return 0
    print("VERDICT=default_pc_was_good_enough")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
