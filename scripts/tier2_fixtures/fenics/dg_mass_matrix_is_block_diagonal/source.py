"""Tier-2 for fenics dg_methods#8: the DG mass matrix is block-DIAGONAL, one
block per cell, so inverting it in an explicit time step is a local operation.
Handing it to a generic sparse solver throws that structure away.

Wrong variant: scipy.sparse.linalg.spsolve on the assembled DG mass matrix,
once per time step, instead of a per-cell solve of the cell blocks.

Measured on 32x32 triangles with DG2 (2048 cells, 6 dofs per cell, 12288 dofs).
The structure is verified before it is exploited: the dofmap really is cell
blocked, the mass matrix has ZERO stored entries outside its cell blocks, and
the implicit DG operator (stiffness plus an interior-facet jump coupling) has
hundreds of thousands of them — so the claim's last sentence, that only the
mass is block-diagonal, is checked too. Then both solvers are timed on the same
right-hand side, best of five runs each, and their answers are compared.
Observed: the generic sparse solve is several times slower than the batched
per-cell solve, which is the operation an explicit DG step actually needs.

Mutation control: T2_MUTATE=1 puts the per-cell solve in the slot where the
generic solver was, so the two timings measure the same work and the speed
difference disappears.
"""
from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402
from dolfinx.fem.petsc import assemble_matrix  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

FCO = {"quadrature_degree": 4}
N, DEGREE, REPEAT = 32, 2, 5


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = fem.functionspace(msh, ("DG", DEGREE))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = (h("+") + h("-")) / 2.0

    M = assemble_matrix(fem.form(u * v * ufl.dx, form_compiler_options=FCO))
    M.assemble()
    K = assemble_matrix(fem.form(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + 10.0 / h_avg * ufl.inner(ufl.jump(u, n), ufl.jump(v, n)) * ufl.dS,
        form_compiler_options=FCO))
    K.assemble()

    bs = V.dofmap.dof_layout.num_dofs
    ncell = msh.topology.index_map(tdim).size_local
    ndof = V.dofmap.index_map.size_global
    print(f"ndofs={ndof} dofs_per_cell={bs} ncells={ncell}")

    def csr(A):
        ai, aj, av = A.getValuesCSR()
        return sp.csr_matrix((av, aj, ai), shape=A.getSize())

    def n_offblock(A):
        coo = csr(A).tocoo()
        same = (coo.row // bs) == (coo.col // bs)
        return int(np.sum(~same))

    dm = V.dofmap.list
    blocked = all(np.array_equal(np.sort(dm[c]),
                                 np.arange(c * bs, (c + 1) * bs))
                  for c in range(ncell))
    print(f"dofmap_is_cell_blocked={blocked}")
    m_off, k_off = n_offblock(M), n_offblock(K)
    print(f"mass_offblock_entries={m_off} implicit_operator_offblock_entries="
          f"{k_off}")
    print(f"mass_matrix_is_block_diagonal={m_off == 0}")
    print(f"implicit_dg_operator_is_not_block_diagonal={k_off > 0}")

    Mc = csr(M).tocsr()
    Mc.sort_indices()
    rows_uniform = bool(np.all(np.diff(Mc.indptr) == bs))
    print(f"every_row_has_one_block_of_entries={rows_uniform}")
    blocks = Mc.data.reshape(ncell, bs, bs)

    rng = np.random.default_rng(0)
    rhs = rng.standard_normal(ndof)
    Mcsc = Mc.tocsc()

    def generic():
        return spla.spsolve(Mcsc, rhs)

    def local():
        return np.linalg.solve(blocks, rhs.reshape(ncell, bs)).reshape(-1)

    slot = local if MUTATE else generic

    def best(fn):
        t_best, out = float("inf"), None
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            out = fn()
            t_best = min(t_best, time.perf_counter() - t0)
        return t_best, out

    t_slot, x_slot = best(slot)
    t_local, x_local = best(local)
    same = bool(np.allclose(x_slot, x_local, rtol=1e-8, atol=1e-10))
    ratio = t_slot / t_local
    print(f"slot_best_ms={t_slot * 1e3:.3f} local_solve_best_ms="
          f"{t_local * 1e3:.3f} ratio={ratio:.2f}")
    print(f"same_answer_both_ways={same}")
    slower = ratio > 2.0
    print(f"generic_sparse_solver_is_slower={slower}")

    if (blocked and m_off == 0 and k_off > 0 and rows_uniform and same
            and slower):
        print("VERDICT=dg_mass_is_block_diagonal_invert_it_per_cell")
        return 0
    print("VERDICT=no_advantage_from_the_block_structure")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
