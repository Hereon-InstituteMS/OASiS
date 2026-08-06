"""Tier-2 for fenics magnetostatics#2: curl-curl needs H(curl) (Nedelec)
elements. Vector Lagrange does not fail loudly - it assembles an ordinary
matrix and solves without complaint - but the whole discrete spectrum is wrong.

Test problem: the 2D cavity eigenproblem (curl u, curl v) = lambda (u, v) on the
unit square with a zero tangential trace, whose exact eigenvalues are
lambda/pi^2 = 1, 1, 2, 4, 4, 5, 5, ... The matrices are restricted to the free
dofs and diagonalised densely, and the gradient kernel (lambda at round-off) is
dropped, so what is compared is the smallest genuinely non-zero eigenvalues.

Observed: N1curl degree 1 on 16x16 gives 0.99807 0.99979 2.00212 3.98288
3.98294 4.98260 5.01511. Vector Lagrange degree 1 gives 0.10753 (8x8), 0.02570
(16x16), 0.00635 (32x32) as its lowest - nothing near a true eigenvalue.
FINDING: the claim says those spurious values DRIFT UPWARD under refinement
(0.108, 0.240, 0.444); measured here the 8x8 value agrees (0.108) but the
sequence drifts DOWNWARD, i.e. the spurious modes collapse onto zero as h falls.
Either way they never approach the true spectrum. Vector Lagrange degree 2 on
16x16 gives 0.56043 0.94579 0.97069 1.02375 1.13483 1.92527 1.98843 - real modes
interleaved with spurious ones, exactly as the claim warns.

Mutation control: T2_MUTATE=1 tests N1curl degree 1 instead, whose eigenvalues
are within a percent of the exact ones.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))
# Dense LAPACK on 2000-dof blocks: one thread is an order of magnitude faster
# here than the default over-subscribed pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

PI2 = np.pi ** 2
EXACT = np.array([1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0])


def element(msh, kind: str, degree: int):
    if kind == "N1curl":
        return basix.ufl.element("N1curl", msh.basix_cell(), degree)
    return basix.ufl.element("Lagrange", msh.basix_cell(), degree,
                             shape=(msh.geometry.dim,))


def spectrum(n: int, kind: str, degree: int, count: int = 7):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = dolfinx.fem.functionspace(msh, element(msh, kind, degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(
        ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx))
    A.assemble()
    M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(
        ufl.inner(u, v) * ufl.dx))
    M.assemble()
    bs = V.dofmap.index_map_bs
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    bdofs = (np.asarray(bdofs)[:, None] * bs + np.arange(bs)[None, :]).ravel()
    free = np.setdiff1d(np.arange(V.dofmap.index_map.size_global * bs), bdofs)
    ix = np.ix_(free, free)
    w = scipy.linalg.eigh(A.convert("dense").getDenseArray()[ix],
                          M.convert("dense").getDenseArray()[ix],
                          eigvals_only=True)
    w = np.sort(w[w > 1e-6]) / PI2
    return w[:count]


def near_true(values: np.ndarray, tol: float) -> np.ndarray:
    return np.array([np.min(np.abs(EXACT - x)) / x < tol for x in values])


def main() -> int:
    kind, degree = ("N1curl", 1) if MUTATE else ("Lagrange", 1)
    print(f"tested_element={kind}{degree}")
    lowest = []
    for n in (8, 16, 32):
        w = spectrum(n, kind, degree)
        lowest.append(float(w[0]))
        print(f"mesh={n}x{n} tested_lambda_over_pi2="
              + " ".join(f"{x:.5f}" for x in w))
    ref = spectrum(16, "N1curl", 1)
    print("mesh=16x16 n1curl_reference_lambda_over_pi2="
          + " ".join(f"{x:.5f}" for x in ref))
    lag2 = spectrum(16, "Lagrange", 2)
    print("mesh=16x16 vector_lagrange_degree_2_lambda_over_pi2="
          + " ".join(f"{x:.5f}" for x in lag2))

    tested_16 = spectrum(16, kind, degree)
    hits = near_true(tested_16, 0.05)
    ref_hits = near_true(ref, 0.05)
    l2_hits = near_true(lag2, 0.05)
    l2_miss = ~near_true(lag2, 0.20)
    drift_away = all(
        abs(lowest[i + 1] - 1.0) > abs(lowest[i] - 1.0)
        for i in range(len(lowest) - 1))
    print(f"tested_lowest_by_refinement={[round(x, 5) for x in lowest]}")
    print(f"tested_eigenvalues_near_a_true_one={int(hits.sum())}")
    print(f"no_tested_eigenvalue_is_near_a_true_one={not hits.any()}")
    print(f"tested_lowest_moves_away_from_one_under_refinement={drift_away}")
    print(f"n1curl_reference_matches_the_exact_spectrum={bool(ref_hits.all())}")
    print(f"vector_lagrange_degree_2_interleaves_spurious_and_real="
          f"{bool(l2_hits.any() and l2_miss.any())}")
    if (not hits.any()) and drift_away and ref_hits.all() \
            and l2_hits.any() and l2_miss.any():
        print("VERDICT=vector_lagrange_curl_curl_spectrum_is_entirely_wrong")
        return 0
    print("VERDICT=tested_element_reproduces_the_cavity_spectrum")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
