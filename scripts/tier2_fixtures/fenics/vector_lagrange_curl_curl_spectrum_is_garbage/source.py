"""Tier-2 for fenics magnetostatics#2: curl-curl needs H(curl) (Nedelec)
elements. Vector Lagrange does not fail loudly -- the form compiles, the matrix
assembles, the eigensolver returns finite numbers -- but the whole discrete
spectrum is wrong.

Wrong variant: the 2D Maxwell cavity eigenproblem
(curl E, curl v) = lambda (E, v) with a zero tangential trace on the unit
square, discretised with vector Lagrange degree 1. Exact lambda/pi^2 =
1, 1, 2, 4, 4, 5, 5. Both matrices are formed densely and the pencil is solved
with LAPACK, so no iterative solver or spectral transform can be blamed. The
exact kernel of curl (gradients) is filtered out by dropping lambda/pi^2 < 1e-6.

Observed on dolfinx 0.10.0: N1curl degree 1 at 32x32 returns 0.99952, 0.99995,
2.00053, 3.99572, 3.99572, 4.99564, 5.00382 -- correct to 3-4 digits. Vector
Lagrange degree 1 on the same meshes returns 0.04021 / 0.00787 / 0.00243 as its
lowest non-kernel eigenvalue at 8x8 / 16x16 / 32x32: none of its modes is
anywhere near a true eigenvalue.

FINDING, recorded against the claim text: the claim says the vector Lagrange
values DRIFT UPWARD under refinement (0.108, 0.240, 0.444). Measured here with
the procedure that reproduces the claim's N1curl numbers exactly, they drift
DOWNWARD -- the spurious modes collapse toward zero as h decreases, which is
the classic spurious-mode picture. Degree 2 vector Lagrange is not a fix
either: at 16x16 it returns 1.76809, 1.98995, 2.20961, 2.57983, ... , i.e. a
couple of nearly-right modes interleaved with spurious ones, which is worse
because it looks plausible.

Mutation control: T2_MUTATE=1 puts N1curl in the space under test; its modes
then sit on the exact spectrum and the garbage-spectrum signal disappears.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

PI2 = np.pi ** 2
EXACT = np.array([1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0])
LEVELS = (8, 16, 32)


def element(msh, family: str, degree: int):
    if family == "n1curl":
        return basix.ufl.element("N1curl", msh.basix_cell(), degree)
    return basix.ufl.element("Lagrange", msh.basix_cell(), degree,
                             shape=(msh.geometry.dim,))


def spectrum(n: int, family: str, degree: int, nkeep: int = 7):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = dolfinx.fem.functionspace(msh, element(msh, family, degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    K = dolfinx.fem.assemble_matrix(dolfinx.fem.form(
        ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx))
    M = dolfinx.fem.assemble_matrix(dolfinx.fem.form(ufl.inner(u, v) * ufl.dx))
    K.scatter_reverse()
    M.scatter_reverse()
    Kd, Md = K.to_dense(), M.to_dense()
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    free = np.setdiff1d(np.arange(Kd.shape[0]), bdofs)
    w = sla.eigh(Kd[np.ix_(free, free)], Md[np.ix_(free, free)],
                 eigvals_only=True)
    lam = np.sort(w) / PI2
    return lam[lam > 1e-6][:nkeep], len(free)


def near_exact(vals: np.ndarray, rel: float) -> int:
    return int(sum(np.min(np.abs(EXACT - x) / EXACT) < rel for x in vals))


def matrix_shape_3d(family: str):
    msh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = dolfinx.fem.functionspace(msh, element(msh, family, 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    A = dolfinx.fem.assemble_matrix(dolfinx.fem.form(
        ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx))
    A.scatter_reverse()
    d = A.to_dense()
    nnz = int(np.count_nonzero(d))
    rows_all_zero = int(np.sum(np.all(d == 0.0, axis=1)))
    return d.shape[0], nnz, float(np.abs(d).max()), rows_all_zero


def main() -> int:
    fam = "n1curl" if MUTATE else "lagrange"
    print(f"space_under_test={fam}_degree_1")

    # the form compiles for every family: the mistake is not caught here
    compiled = []
    for f in ("n1curl", "lagrange"):
        msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        V = dolfinx.fem.functionspace(msh, element(msh, f, 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        try:
            dolfinx.fem.form(ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx)
            compiled.append(True)
        except Exception as exc:
            compiled.append(False)
            print(f"curl_curl_form_on_{f}_failed {type(exc).__name__}: {exc}")
    print(f"curl_curl_form_compiles_on_every_family={all(compiled)}")

    lowest = []
    for n in LEVELS:
        vals, nfree = spectrum(n, fam, 1)
        lowest.append(float(vals[0]))
        print(f"level={n}x{n} {fam}1 free_dofs={nfree} "
              f"lambda_over_pi2={np.round(vals, 5).tolist()}")
        print(f"  modes_within_5_percent_of_exact={near_exact(vals, 0.05)}"
              f" of {len(vals)}")

    ref, nref = spectrum(LEVELS[-1], "n1curl", 1)
    print(f"reference={LEVELS[-1]}x{LEVELS[-1]} n1curl1 free_dofs={nref} "
          f"lambda_over_pi2={np.round(ref, 5).tolist()}")
    ref_err = float(np.max(np.abs(ref - EXACT) / EXACT))
    print(f"reference_max_relative_error={ref_err:.5f}")
    print(f"reference_n1curl_matches_exact_within_1_percent={ref_err < 0.01}")

    vals_fine, _ = spectrum(LEVELS[-1], fam, 1)
    hits = near_exact(vals_fine, 0.05)
    print(f"under_test_has_no_mode_within_5_percent_of_the_exact_spectrum="
          f"{hits == 0}")
    collapse = all(lowest[i + 1] < 0.7 * lowest[i]
                   for i in range(len(lowest) - 1))
    print(f"under_test_lowest_modes={[round(x, 5) for x in lowest]}")
    print(f"under_test_spurious_modes_collapse_toward_zero={collapse}")

    p2, _ = spectrum(16, "lagrange", 2)
    p2_hits = near_exact(p2, 0.02)
    print(f"vector_lagrange_degree_2_at_16x16="
          f"{np.round(p2, 5).tolist()} modes_within_2_percent={p2_hits}")
    inter = 0 < p2_hits < len(p2) and abs(p2[0] - 1.0) > 0.1
    print(f"vector_lagrange_degree_2_interleaves_real_and_spurious={inter}")

    ordinary = True
    for f in ("lagrange", "n1curl"):
        ndof, nnz, mx, zrows = matrix_shape_3d(f)
        print(f"cube_4x4x4 {f}1 dofs={ndof} nonzeros={nnz} "
              f"max_abs_entry={mx:.4f} all_zero_rows={zrows}")
        ordinary &= (nnz > 0 and np.isfinite(mx) and mx > 0.5 and zrows == 0)
    print(f"both_3d_matrices_look_ordinary={ordinary}")

    if (all(compiled) and ref_err < 0.01 and hits == 0 and collapse
            and inter and ordinary):
        print("VERDICT=vector_lagrange_spectrum_is_wrong_while_n1curl_is_right")
        return 0
    print("VERDICT=space_under_test_reproduced_the_exact_spectrum")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
