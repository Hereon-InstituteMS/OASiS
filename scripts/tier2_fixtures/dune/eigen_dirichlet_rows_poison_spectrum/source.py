"""Tier-2: the two ways a FEM eigenvalue computation goes wrong here.

  eigenvalue#1   A galerkin scheme's constrained rows are IDENTITY rows.
                 Feeding that matrix to a generalised eigensolver adds
                 exactly one extra eigenvalue per constrained dof, and
                 makes the matrix NON-SYMMETRIC, which silently violates
                 what eigsh assumes. Deleting the rows is not enough —
                 the COLUMNS have to go too; the interior submatrices
                 then reproduce the analytic pi^2 (m^2 + n^2).

                 This fixture also FALSIFIES two specifics the catalog
                 used to state, and prints the counter-evidence rather
                 than just omitting them: the extra eigenvalues are NOT
                 at 1/M_ii, and shift-invert at sigma=0 does NOT return
                 them first. See the fixture _comment.

  eigenvalue#3   eigsh without a shift returns the wrong end of the
                 spectrum: the default which='LM' gives the O(1/h^2)
                 mesh artefacts. sigma=0.0 with which='LM'
                 (shift-invert) is the right call.

24x24 P1: 625 dofs, 96 on the boundary, 529 interior. The full spectra
are computed densely, because the whole point is to COUNT modes and
locate them, which a Krylov method cannot do.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                           # noqa: E402
from dune.fem.space import lagrange                            # noqa: E402
from dune.fem.scheme import galerkin                           # noqa: E402
from dune.ufl import DirichletBC                               # noqa: E402
import dune.fem as dfem                                        # noqa: E402
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, # noqa: E402
                 dot, grad, dx, conditional, sqrt)


def main() -> int:
    fail: list[str] = []
    n = 24
    gridView = structuredGrid([0, 0], [1, 1], [n, n])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    stiff = dot(grad(u), grad(v)) * dx
    mass = u * v * dx

    A = dfem.assemble(stiff).as_numpy
    M = dfem.assemble(mass).as_numpy
    print(f"n_dofs={space.size}")
    if space.size != (n + 1) ** 2:
        fail.append(f"expected {(n + 1) ** 2} dofs, got {space.size}")

    x = SpatialCoordinate(space)
    tol = 1e-8
    onb = conditional(
        sqrt((x[0] - 0.5) ** 2) > 0.5 - tol, 1.0,
        conditional(sqrt((x[1] - 0.5) ** 2) > 0.5 - tol, 1.0, 0.0))
    bnd = np.array(space.interpolate(onb, name="bnd").as_numpy) > 0.5
    interior = ~bnd
    n_bnd, n_int = int(bnd.sum()), int(interior.sum())
    print(f"n_boundary_dofs={n_bnd}")
    print(f"n_interior_dofs={n_int}")
    if n_bnd != 4 * n:
        fail.append(f"the boundary marker found {n_bnd} dofs; a {n}x{n} "
                    f"Q1 grid has {4 * n}")

    # ── the constrained matrix ──────────────────────────────────────
    scheme = galerkin([stiff == 1.0 * v * dx, DirichletBC(space, 0)],
                      solver="cg")
    Ac = dfem.operator.linear(scheme).as_numpy
    diag = Ac.diagonal()
    row_sums = np.array(abs(Ac).sum(axis=1)).ravel()
    identity_rows = bool(np.allclose(diag[bnd], 1.0)
                         and np.allclose(row_sums[bnd], 1.0))
    print(f"constrained_rows_are_identity_rows={identity_rows}")
    if not identity_rows:
        fail.append(f"the constrained rows are not identity rows "
                    f"(diagonal range {diag[bnd].min()}.."
                    f"{diag[bnd].max()}, row-abs-sum range "
                    f"{row_sums[bnd].min()}..{row_sums[bnd].max()}); the "
                    f"whole claim rests on them being identity rows")

    asym = float(abs(Ac - Ac.T).max())
    print(f"constrained_matrix_asymmetry={asym:.6f}")
    print(f"constrained_matrix_is_not_symmetric={asym > 1e-12}")
    if asym <= 1e-12:
        fail.append("the constrained matrix is symmetric, so the claim "
                    "that identity rows break the symmetry eigsh "
                    "assumes does not hold")

    # ── count and LOCATE the extra modes ────────────────────────────
    lam_c = np.sort(sla.eigvals(Ac.toarray(), M.toarray()).real)
    Ai = A[interior][:, interior].toarray()
    Mi = M[interior][:, interior].toarray()
    lam_i = sla.eigh(Ai, Mi, eigvals_only=True)
    lam_i = np.sort(lam_i)
    print(f"constrained_spectrum_size={lam_c.size}")
    print(f"interior_spectrum_size={lam_i.size}")
    print(f"extra_mode_count_expected={n_bnd}")
    if lam_c.size - lam_i.size != n_bnd:
        fail.append(f"the constrained pencil has {lam_c.size} modes and "
                    f"the physical one {lam_i.size}; the difference "
                    f"{lam_c.size - lam_i.size} should equal the "
                    f"{n_bnd} constrained dofs")

    # Greedily match every physical eigenvalue to its nearest unused
    # constrained one; what is left over is the pollution.
    remaining = list(range(lam_c.size))
    for val in lam_i:
        j = min(remaining, key=lambda k: abs(lam_c[k] - val))
        remaining.remove(j)
    extra = np.sort(lam_c[remaining])
    print(f"extra_modes_found={extra.size}")
    print(f"extra_modes_min={extra.min():.4f}")
    print(f"extra_modes_max={extra.max():.4f}")
    print(f"first_physical_eigenvalue={lam_i[0]:.4f}")
    print(f"extra_modes_are_above_the_physical_bottom="
          f"{extra.min() > 10 * lam_i[0]}")
    if extra.size != n_bnd:
        fail.append(f"matching left {extra.size} unmatched constrained "
                    f"modes, not {n_bnd}")

    # FALSIFICATION 1: they are not at 1/M_ii.
    inv_mii = np.unique(np.round(1.0 / M.diagonal()[bnd], 6))
    near = int(sum(1 for e in extra
                   if float(np.min(np.abs(inv_mii - e) / inv_mii)) < 1e-3))
    print(f"one_over_Mii_boundary_values={inv_mii.tolist()}")
    print(f"extra_modes_at_one_over_Mii={near}")
    print(f"one_over_Mii_prediction_is_false={near == 0}")
    if near != 0:
        fail.append(f"{near} extra modes DO sit at 1/M_ii; the catalog's "
                    f"retracted prediction would then have been right "
                    f"and this fixture's correction is wrong")

    # FALSIFICATION 2: sigma=0 returns physics, not pollution.
    k = 6
    shift_c = np.sort(eigsh(Ac.tocsc(), k=k, M=M.tocsc(), sigma=0.0,
                            which="LM", return_eigenvectors=False))
    analytic = np.sort([np.pi ** 2 * (a ** 2 + b ** 2)
                        for a in range(1, 6) for b in range(1, 6)])[:k]
    rel_c = np.abs(shift_c - analytic) / analytic
    print(f"sigma0_on_constrained={np.array2string(shift_c[:4], precision=4)}")
    print(f"analytic_smallest={np.array2string(analytic[:4], precision=4)}")
    print(f"sigma0_max_relative_error={rel_c.max():.2e}")
    print(f"sigma0_returns_physics_not_pollution={rel_c.max() < 1e-2}")
    if rel_c.max() >= 1e-2:
        fail.append(f"sigma=0 on the constrained pencil returned values "
                    f"{shift_c} that are not the physical modes; the "
                    f"correction this fixture records would be wrong")

    # ── the fix: delete rows AND columns ───────────────────────────
    clean = lam_i[:k]
    rel_all = np.abs(clean - analytic) / analytic
    # The accuracy gate is the lowest FOUR modes. Q1 discretisation
    # error grows with mode number — measured 1.18e-02 by mode 6 on this
    # grid — and that is the discretisation, not the pollution this
    # fixture is about.
    rel = rel_all[:4]
    print(f"interior_submatrix_shape={Ai.shape}")
    print(f"clean_smallest={np.array2string(clean[:4], precision=4)}")
    print(f"clean_max_relative_error_lowest4={rel.max():.2e}")
    print(f"clean_max_relative_error_lowest{k}={rel_all.max():.2e}")
    print(f"row_and_column_deletion_recovers_analytic={rel.max() < 1e-2}")
    if rel.max() >= 1e-2:
        fail.append(f"the interior submatrices gave a max relative "
                    f"error of {rel.max():.2e} against pi^2(m^2+n^2) "
                    f"over the lowest 4 modes")

    # deleting only the rows leaves a non-square matrix
    rows_only = A[interior]
    print(f"rows_only_shape={rows_only.shape}")
    print(f"rows_only_is_not_square="
          f"{rows_only.shape[0] != rows_only.shape[1]}")
    if rows_only.shape[0] == rows_only.shape[1]:
        fail.append("deleting only the rows left a square matrix, so "
                    "the 'rows AND columns' half of the claim cannot be "
                    "demonstrated")

    # the degenerate pair stays exactly degenerate on the clean problem
    split = float(abs(lam_i[2] - lam_i[1]))
    print(f"degenerate_pair_split_clean={split:.3e}")
    print(f"clean_problem_keeps_the_degeneracy={split < 1e-8}")
    if split >= 1e-8:
        fail.append(f"the analytically degenerate (1,2)/(2,1) pair "
                    f"split by {split:.3e} on the interior submatrices")

    # ── eigenvalue#3: the shift ────────────────────────────────────
    largest = np.sort(eigsh(Ai, k=4, M=Mi, which="LM",
                            return_eigenvectors=False))
    h = 1.0 / n
    print(f"no_shift_LM_smallest_returned={largest.min():.4e}")
    print(f"one_over_h_squared={1.0 / h ** 2:.4e}")
    print(f"no_shift_LM_returns_mesh_artefacts="
          f"{largest.min() > 1.0 / h ** 2}")
    if largest.min() <= 1.0 / h ** 2:
        fail.append(f"which='LM' without sigma returned "
                    f"{largest.min():.4e}, below 1/h^2 = "
                    f"{1.0 / h ** 2:.4e}; the claim is that it returns "
                    f"the O(1/h^2) end")

    if not fail:
        print("dune_eigenvalue_route_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
