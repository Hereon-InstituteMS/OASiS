"""Tier-2 for fenics eigenvalue#4: the two corrections to the Dirichlet mass
matrix recipe.

(1) `diag=0.0` on B does NOT work with the default spectral transform. The
default EPS transform is a plain shift, which factorises B itself, and the
zeroed constrained rows are an exact zero pivot. Wrong variant: B assembled
with bcs and diag=0.0 and no ST set. Right variant: the same matrices with
eps.getST().setType(SLEPc.ST.Type.SINVERT).

(2) Assembling B with `bcs=[]` is NOT a clean substitute. With a CONSISTENT
mass matrix the boundary-to-interior blocks are non-zero, so it solves a
different pencil.

Observed on dolfinx 0.10.0 / slepc4py 3.24.3, 8x8 and 16x16 P1 unit square,
reference = dense scipy.linalg.eigvalsh of the interior blocks:

  * the plain-shift run stops inside STSetUp_Shift -> KSPSetUp -> PCSetUp_LU ->
    MatLUFactorNumeric_SeqAIJ with "Zero pivot in LU factorization" and
    "Zero pivot row 0 value 0. tolerance 2.22045e-14", and reaches Python as
    SystemError "<cyfunction EPS.solve at 0x...> returned a result with an
    exception set"; adding SINVERT makes the identical run succeed;
  * B with bcs=[] is off by 5.848e-02 relative at 8x8 and 1.163e-03 at 16x16 —
    wrong in the second significant digit on the coarse mesh and shrinking
    under refinement, which is exactly why the error is easy to miss — while B
    with diag=0.0 matches the reduced problem to 1.3e-14 and 7.6e-14 on the
    same two meshes.

Mutation control: T2_MUTATE=1 gives the primary solve SINVERT and uses the
diag=0.0 mass matrix in place of the bcs=[] one, so neither symptom appears and
the fixture loses its own expectations.
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

import scipy.linalg as sla  # noqa: E402
from petsc4py import PETSc  # noqa: E402
from slepc4py import SLEPc  # noqa: E402

MESHES = (8, 16)
NEV = 6
NCMP = 5


def build(n):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    m = dolfinx.fem.form(u * v * ufl.dx)
    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = np.asarray(dolfinx.fem.locate_dofs_topological(V, fdim, bfacets))
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), dofs, V)
    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    A_free = dolfinx.fem.petsc.assemble_matrix(a)
    A_free.assemble()
    B_free = dolfinx.fem.petsc.assemble_matrix(m)
    B_free.assemble()
    Ad = A_free.copy().convert("dense").getDenseArray().copy()
    Bd = B_free.copy().convert("dense").getDenseArray().copy()
    free = np.setdiff1d(np.arange(ndof), dofs)
    reduced = np.sort(sla.eigvalsh(Ad[np.ix_(free, free)],
                                   Bd[np.ix_(free, free)]))

    A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    B_zero = dolfinx.fem.petsc.assemble_matrix(m, bcs=[bc], diag=0.0)
    B_zero.assemble()
    return msh, A, B_zero, B_free, reduced, free.size


def solve(msh, A, B, sinvert):
    eps = SLEPc.EPS().create(msh.comm)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(NEV, PETSc.DECIDE)
    eps.setTolerances(1.0e-11, 3000)
    eps.setTarget(0.0)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    if sinvert:
        eps.getST().setType(SLEPc.ST.Type.SINVERT)
    eps.solve()
    n = eps.getConverged()
    return np.array(sorted(eps.getEigenvalue(i).real for i in range(n)))


def main() -> int:
    msh, A, B_zero, B_free, reduced, nfree = build(MESHES[0])

    # (1) the transform.
    caught = ""
    try:
        solve(msh, A, B_zero, sinvert=MUTATE)
        print("primary_transform_solved=True")
    except Exception as exc:
        caught = f"{type(exc).__name__}: {exc}"
        print(f"primary_transform_solved=False caught={caught}")
        # The PETSc/SLEPc trace is on the chained petsc4py error, not on the
        # SystemError that Cython hands back.
        origin = exc.__cause__ or exc.__context__
        if origin is not None:
            print("petsc_error_chain>>>")
            print(str(origin))
            print("<<<petsc_error_chain")
    vals_ok = solve(msh, A, B_zero, sinvert=True)
    print(f"with_sinvert_lowest={np.round(vals_ok[:3], 7).tolist()}")

    # (2) the pencil.
    diffs_empty, diffs_zero = [], []
    for n in MESHES:
        msh_n, A_n, Bz_n, Bf_n, red_n, _ = build(n)
        v_empty = solve(msh_n, A_n, Bf_n if not MUTATE else Bz_n, sinvert=True)
        v_zero = solve(msh_n, A_n, Bz_n, sinvert=True)
        d_e = float(np.max(np.abs(v_empty[:NCMP] - red_n[:NCMP])
                           / red_n[:NCMP]))
        d_z = float(np.max(np.abs(v_zero[:NCMP] - red_n[:NCMP])
                           / red_n[:NCMP]))
        diffs_empty.append(d_e)
        diffs_zero.append(d_z)
        print(f"N={n} primary_B_maxreldiff_vs_reduced={d_e:.3e} "
              f"diag0_B_maxreldiff_vs_reduced={d_z:.3e}")

    failed_without_sinvert = bool(caught) and not MUTATE
    coarse_wrong = diffs_empty[0] > 1.0e-3
    gap_shrinks = diffs_empty[-1] < 0.1 * diffs_empty[0]
    zero_exact = all(d < 1.0e-10 for d in diffs_zero)
    print(f"primary_default_transform_hit_a_zero_pivot={failed_without_sinvert}")
    print(f"primary_mass_matrix_wrong_in_second_significant_digit={coarse_wrong}")
    print(f"primary_error_shrinks_tenfold_under_refinement={gap_shrinks}")
    print(f"diag_zero_matches_reduced_problem_on_every_mesh={zero_exact}")
    if failed_without_sinvert and coarse_wrong and gap_shrinks and zero_exact:
        print("VERDICT=diag_zero_needs_sinvert_and_bcs_empty_is_a_different_pencil")
        return 0
    print("VERDICT=default_transform_and_bcs_empty_were_both_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
