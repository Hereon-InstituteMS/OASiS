"""Tier-2 for fenics mixed_poisson#4: RT(k) pairs with DG(k-1), never DG(k).
Pairing RT(1) with DG(1) makes the saddle-point matrix rank deficient -- the
pressure block is bigger than the flux block can constrain -- and a plain LU
factorisation stops with a zero pivot.

The wrong variant is basix.ufl.mixed_element([RT(1), DG(1)]); the right one is
[RT(1), DG(0)]. Both are assembled and both are factorised here, so the
contrast is measured in-process: a dense SVD counts the null vectors and PETSc
LU is asked to factor the same matrix.

Observed on dolfinx 0.10.0: DG(1) has 96 pressure dofs against 56 RT(1) flux
dofs on a 4x4 unit square, so the divergence block cannot have full row rank;
the 152x152 matrix has 64 null vectors and the pivoting LAPACK LU stops with
"Zero pivot in LU factorization". The RT(1)xDG(0) matrix has full rank and
factors. NOTE PETSc's SPARSE LU does no pivoting at all and trips over the zero
pressure block of BOTH pairs, so the fixture factors dense copies. The "the
pressure error does not converge under refinement" half of the claim is not
measurable separately: the singular system yields no solution to take an error
from.

Mutation control: T2_MUTATE=1 pairs RT(1) with DG(0) in the "bad" slot, the
matrix then has full rank and LU succeeds.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from petsc4py import PETSc  # noqa: E402

N = 4


def assemble_pair(p_degree: int):
    """RT(1) x DG(p_degree) mixed Poisson operator on an N x N unit square."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    S = basix.ufl.element("RT", msh.basix_cell(), 1)
    P = basix.ufl.element("DG", msh.basix_cell(), p_degree)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([S, P]))
    (sig, p) = ufl.TrialFunctions(W)
    (tau, q) = ufl.TestFunctions(W)
    a = (ufl.inner(sig, tau) * ufl.dx
         - p * ufl.div(tau) * ufl.dx
         + q * ufl.div(sig) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    A.assemble()
    V0, _ = W.sub(0).collapse()
    Q0, _ = W.sub(1).collapse()
    n_flux = V0.dofmap.index_map.size_global * V0.dofmap.index_map_bs
    n_pres = Q0.dofmap.index_map.size_global * Q0.dofmap.index_map_bs
    return msh, A, n_flux, n_pres


def rank(A) -> tuple[int, int]:
    dense = A.copy().convert("dense").getDenseArray().copy()
    sv = np.linalg.svd(dense, compute_uv=False)
    return int(np.sum(sv > sv[0] * 1e-10)), dense.shape[0]


def try_lu(msh, A) -> tuple[bool, str]:
    # Dense LAPACK LU, i.e. WITH partial pivoting. PETSc's sparse LU does not
    # pivot, so it trips over the zero pressure block of any saddle-point
    # matrix, well posed or not; that would not separate the two pairs.
    Ad = A.copy().convert("dense")
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(Ad)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setErrorIfNotConverged(True)
    b = Ad.createVecRight()
    b.set(1.0)
    x = Ad.createVecRight()
    try:
        ksp.solve(b, x)
    except Exception as exc:  # noqa: BLE001 - the message is the evidence
        return False, str(exc)
    return True, ""


def main() -> int:
    bad_degree = 0 if MUTATE else 1
    msh_b, A_b, nf_b, np_b = assemble_pair(bad_degree)
    r_b, n_b = rank(A_b)
    ok_b, msg_b = try_lu(msh_b, A_b)

    msh_g, A_g, nf_g, np_g = assemble_pair(0)
    r_g, n_g = rank(A_g)
    ok_g, _ = try_lu(msh_g, A_g)

    print(f"pair_under_test=RT1xDG{bad_degree} flux_dofs={nf_b} "
          f"pressure_dofs={np_b} matrix={n_b} rank={r_b}")
    print(f"reference_pair=RT1xDG0 flux_dofs={nf_g} pressure_dofs={np_g} "
          f"matrix={n_g} rank={r_g}")
    deficient = r_b < n_b
    print(f"dg_k_pressure_dofs_exceed_rt_flux_dofs={np_b > nf_b}")
    print(f"bad_pair_matrix_is_rank_deficient={deficient}")
    print(f"bad_pair_null_vectors={n_b - r_b}")
    print(f"good_pair_matrix_has_full_rank={r_g == n_g}")
    print(f"good_pair_lu_succeeded={ok_g}")
    print(f"bad_pair_lu_succeeded={ok_b}")
    if msg_b:
        print("--- PETSc error from the LU factorisation of the bad pair ---")
        print(msg_b)
        print("--- end PETSc error ---")
    if deficient and not ok_b and r_g == n_g and ok_g:
        print("VERDICT=rt_paired_with_dg_k_is_singular")
        return 0
    print("VERDICT=rt_paired_with_dg_k_solved_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
