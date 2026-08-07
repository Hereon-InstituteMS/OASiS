"""Tier-2: dune.fem.assemble — what it returns, and what it ignores.

  eigenvalue#0                        there is no eigen solver in
                                      dune.fem at all and `import
                                      dune.fem.solver` raises
                                      ModuleNotFoundError, so a spectral
                                      problem must go assemble ->
                                      .as_numpy -> scipy. assemble()
                                      returns a LinearOperator whose
                                      only conversion attribute is
                                      .as_numpy.
  eigenvalue#2                        .as_numpy is ALREADY a scipy
                                      csr_matrix, so a .tocsr() is a
                                      no-op; fancy indexing works
                                      directly. (An earlier revision of
                                      the catalog said COO.)
  _general assemble_measured.Signal   the same no-eigensolver route.
  _general bc_argument_is_silently_
  ignored                             assemble(form, bc) accepts a
                                      DirichletBC and silently ignores
                                      it: same nnz, max|A1-A2| exactly
                                      0.0.

One assembled matrix answers all four, so they share a fixture.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 takes the "with bc" matrix from the
SCHEME's linear operator — the object that really does apply the
constraints — instead of from assemble(a, bc). The bc is then not
ignored, assemble_with_bc_maxdiff stops being 0.0 and
bc_argument_silently_ignored reads False, so both expectations
disappear. No new form is compiled: the scheme is the one this fixture
already builds at the end.
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                           # noqa: E402
from dune.fem.space import lagrange                            # noqa: E402
from dune.ufl import DirichletBC                               # noqa: E402
import dune.fem as dfem                                        # noqa: E402
from ufl import (TrialFunction, TestFunction,                   # noqa: E402
                 dot, grad, dx)


def main() -> int:
    fail: list[str] = []

    # ── eigenvalue#0: there is no eigensolver, and no solver module ──
    for name in ("eig", "eigs", "eigsh", "eigenvalue", "eigenvalues",
                 "eigenSolver"):
        present = hasattr(dfem, name)
        print(f"dune_fem_has_{name}={present}")
        if present:
            fail.append(f"dune.fem.{name} exists; the claim that there "
                        f"is no eigenvalue entry point is stale")
    try:
        importlib.import_module("dune.fem.solver")
        print("import_dune_fem_solver_raises=False")
        fail.append("dune.fem.solver is importable; the claim is "
                    "ModuleNotFoundError")
    except ImportError as exc:
        print(f"import_dune_fem_solver_raises={type(exc).__name__}")

    # ── what assemble() returns ─────────────────────────────────────
    gridView = structuredGrid([0, 0], [1, 1], [16, 16])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    a = dot(grad(u), grad(v)) * dx

    A = dfem.assemble(a)
    print(f"assemble_returns={type(A).__name__}")
    conv = sorted(n for n in dir(A)
                  if "numpy" in n or "petsc" in n or "istl" in n)
    print(f"conversion_attributes={conv}")
    if type(A).__name__ != "LinearOperator":
        fail.append(f"assemble() returned {type(A).__name__}, not the "
                    f"LinearOperator the claim names")
    if conv != ["as_numpy"]:
        fail.append(f"the conversion attributes are {conv}; the claim "
                    f"is that .as_numpy is the only one")

    # ── eigenvalue#2: it is CSR already ─────────────────────────────
    M = A.as_numpy
    print(f"as_numpy_type={type(M).__name__}")
    print(f"as_numpy_format={M.format}")
    print(f"tocsr_is_a_noop={M.tocsr() is M}")
    print(f"as_numpy_shape={M.shape} nnz={M.nnz}")
    if M.format != "csr":
        fail.append(f"as_numpy is in {M.format} format, not csr")
    if M.tocsr() is not M:
        fail.append("tocsr() returned a different object, so it is not "
                    "the no-op the claim says it is")
    if M.shape != (space.size, space.size):
        fail.append(f"the matrix is {M.shape}, expected "
                    f"({space.size}, {space.size})")

    # fancy indexing works with no conversion at all
    mask = np.zeros(M.shape[0], dtype=bool)
    mask[: M.shape[0] // 2] = True
    sub = M[mask][:, mask]
    sub_via_tocsr = M.tocsr()[mask][:, mask]
    same = abs(sub - sub_via_tocsr).max() if sub.nnz else 0.0
    print(f"fancy_index_shape={sub.shape}")
    print(f"fancy_index_type={type(sub).__name__}")
    print(f"fancy_index_matches_tocsr_route={float(same) == 0.0}")
    if not sp.issparse(sub) or float(same) != 0.0:
        fail.append("fancy indexing .as_numpy directly disagreed with "
                    "the .tocsr() route; the claim is that they are the "
                    "same because .tocsr() does nothing")

    # ── the bc argument is accepted and ignored ─────────────────────
    from dune.fem.scheme import galerkin
    scheme = galerkin([a == 1.0 * v * dx, DirichletBC(space, 0)],
                      solver="cg")
    if MUTATE:
        # The pathology removed: take the matrix from the object that
        # DOES honour the constraint, so the bc is no longer ignored.
        print("mutation=the_with_bc_matrix_comes_from_the_scheme_"
              "which_really_constrains")
        N = dfem.operator.linear(scheme).as_numpy
    else:
        B = dfem.assemble(a, DirichletBC(space, [0]))
        N = B.as_numpy
    diff = abs(M - N)
    maxdiff = float(diff.max()) if diff.nnz else 0.0
    print(f"assemble_with_bc_accepted=True")
    print(f"assemble_with_bc_nnz={N.nnz} without_bc_nnz={M.nnz}")
    print(f"assemble_with_bc_maxdiff={maxdiff}")
    print(f"bc_argument_silently_ignored="
          f"{N.nnz == M.nnz and maxdiff == 0.0}")
    if not (N.nnz == M.nnz and maxdiff == 0.0):
        fail.append(f"assemble(form, bc) now CHANGES the matrix (nnz "
                    f"{N.nnz} vs {M.nnz}, max|diff| {maxdiff}); the "
                    f"claim that the bc argument is silently ignored is "
                    f"no longer true and the knowledge must be updated")

    # …and the scheme, which is where constraints really live, does
    # change the matrix. Without this the fixture cannot tell 'ignored'
    # from 'this bc happens to be a no-op'.
    S = dfem.operator.linear(scheme).as_numpy
    sdiff = abs(M - S)
    smax = float(sdiff.max()) if sdiff.nnz else 0.0
    print(f"scheme_matrix_differs_from_raw={smax > 0.0}")
    print(f"scheme_matrix_maxdiff={smax:.6f}")
    if smax <= 0.0:
        fail.append("the scheme's matrix is identical to the raw "
                    "Galerkin matrix, so this fixture cannot show that "
                    "the SCHEME is what applies constraints")

    if not fail:
        print("dune_assemble_semantics_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
