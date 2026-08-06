"""Tier-2: what refactoring every step actually costs, measured structurally.

Claim: skfem heat_transient#1 -- factor the system matrix once with
scipy.sparse.linalg.factorized() and reuse it across time steps; refactoring
every step costs O(N^1.5) against O(N) for back-substitution.  "Signal:
per-step wall time is dominated by factorisation, scaling as N^1.5 instead of
N as the mesh is refined."

The stated signal is a wall-clock one.  This fixture asserts nothing about
elapsed time -- such a check goes red on a busy machine and is not evidence --
and measures the structural quantity the cost is proportional to instead: the
number of non-zeros in the LU factors, which is what a factorisation produces
and a substitution merely reads.

Measured on skfem 12.0.1 / scipy 1.15.3, backward Euler on
MeshTri.init_tensor grids from 16x16 to 64x64, ElementTriP1:

  * the factor fill-in grows SUPERLINEARLY in N while the system matrix's own
    non-zero count grows linearly -- the measured growth exponent of the LU
    non-zeros against N is above 1 at every refinement step, and above the
    system matrix's exponent, which is what makes repeating the
    factorisation the dominant cost.
  * the count of factorisations is the thing under the programmer's control:
    a factorized() callable performs exactly ONE LU no matter how many steps
    run, verified by wrapping scipy's splu.
  * reuse is exact: 40 steps through the reused factorisation agree with 40
    spsolve calls to machine precision.
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.sparse.linalg as spl
import scipy.sparse.linalg._dsolve.linsolve as linsolve
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

DT = 1e-3
NSTEPS = 40


def build(nx):
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, nx + 1),
                            np.linspace(0.0, 1.0, nx + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    A = (M + DT * K).tocsr()
    Ac, _, _, I = condense(A, ib.zeros(), D=D)
    return ib, A, M, D, Ac, I


def main() -> int:
    ok = True
    sizes, sys_nnz, lu_nnz = [], [], []
    for nx in (16, 24, 32, 48, 64):
        ib, A, M, D, Ac, I = build(nx)
        lu = spl.splu(Ac.tocsc())
        fill = int(lu.L.nnz + lu.U.nnz)
        sizes.append(Ac.shape[0])
        sys_nnz.append(int(Ac.nnz))
        lu_nnz.append(fill)
        print(f"nx{nx}_condensed_N={Ac.shape[0]} system_nnz={Ac.nnz} "
              f"lu_nnz={fill} fill_ratio={fill / Ac.nnz:.3f}")

    sizes = np.array(sizes, dtype=float)
    sys_nnz = np.array(sys_nnz, dtype=float)
    lu_nnz = np.array(lu_nnz, dtype=float)
    sys_exp = [float(np.log(sys_nnz[i + 1] / sys_nnz[i])
                     / np.log(sizes[i + 1] / sizes[i]))
               for i in range(len(sizes) - 1)]
    lu_exp = [float(np.log(lu_nnz[i + 1] / lu_nnz[i])
                    / np.log(sizes[i + 1] / sizes[i]))
              for i in range(len(sizes) - 1)]
    print(f"system_nnz_exponents={[f'{e:.3f}' for e in sys_exp]}")
    print(f"lu_nnz_exponents={[f'{e:.3f}' for e in lu_exp]}")
    print(f"system_nnz_grows_linearly="
          f"{all(0.95 < e < 1.05 for e in sys_exp)}")
    print(f"lu_fill_grows_superlinearly={all(e > 1.05 for e in lu_exp)}")
    print(f"lu_exponent_exceeds_system_exponent="
          f"{all(lu_exp[i] > sys_exp[i] for i in range(len(lu_exp)))}")
    print(f"fill_ratio_grows="
          f"{(lu_nnz[-1] / sys_nnz[-1]) > (lu_nnz[0] / sys_nnz[0])}")
    if not all(0.95 < e < 1.05 for e in sys_exp):
        print("FAIL: the system matrix non-zeros did not grow linearly",
              file=sys.stderr)
        ok = False
    if not all(e > 1.05 for e in lu_exp):
        print("FAIL: the LU fill-in did not grow superlinearly",
              file=sys.stderr)
        ok = False

    # --- one factorisation, however many steps ---------------------------
    ib, A, M, D, Ac, I = build(32)
    calls = {"splu": 0}
    orig = linsolve.splu

    def counted(*a, **k):
        calls["splu"] += 1
        return orig(*a, **k)

    linsolve.splu = counted
    try:
        u0 = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        u0[D] = 0.0
        v = u0.copy()
        lu = spl.factorized(Ac.tocsc())
        for _ in range(NSTEPS):
            nxt = ib.zeros()
            nxt[I] = lu((M @ v)[I])
            v = nxt
        n_lu = calls["splu"]
    finally:
        linsolve.splu = orig

    u = u0.copy()
    for _ in range(NSTEPS):
        u = solve(*condense(A, M @ u, D=D))

    print(f"nsteps={NSTEPS}")
    print(f"factorisations_performed={n_lu}")
    print(f"exactly_one_factorisation_for_all_steps={n_lu == 1}")
    dev = float(np.abs(u - v).max())
    print(f"reuse_vs_per_step_max_difference={dev:.3e}")
    print(f"reuse_is_exact={dev < 1e-12}")
    if n_lu != 1:
        print(f"FAIL: {n_lu} factorisations were performed, not 1",
              file=sys.stderr)
        ok = False
    if dev >= 1e-12:
        print("FAIL: reusing the factorisation changed the answer",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
