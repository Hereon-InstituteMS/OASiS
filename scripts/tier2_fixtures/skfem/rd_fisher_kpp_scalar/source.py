"""Tier-2: Fisher-KPP is scalar — a block 2x2 Jacobian doubles the DOFs, quadruples the nnz, and contaminates u.

Claim: skfem reaction_diffusion#8 — Fisher-KPP, du/dt = D*lap(u) + r*u*(1-u), is
a SCALAR equation with no coupling block. Building a block-2x2 Jacobian for it
yields a sparse matrix double the expected size, the wasted linear-solve cost
shows in scipy.sparse.linalg.spsolve wall time, and the solution develops
spurious cross-diffusion. The correct discretisation is a single Newton solve on
M + dt*K - dt*M_r with M_r the mass matrix weighted by r*(1-2u^n).

Wrong variant: the same problem set up as a two-field system the way a
Schnakenberg template is written — a second field v that is meant to be a
redundant copy of u, reactions f_u = r*u*(1-v) and f_v = r*v*(1-u), and all four
Jacobian blocks assembled.
Right variant: the scalar Newton solve the claim describes.

Observed on scikit-fem 12.0.1 (2026-08-06), 16x16 ElementQuad1, D=1e-2, r=2,
backward Euler dt=0.05, 40 steps, from a 0.1-amplitude Gaussian bump:

  scalar      matrix 289x289   nnz 2401   mean spsolve 0.89 ms
  block       matrix 578x578   nnz 9604   mean spsolve 2.06 ms

The DOF count doubles as the claim says; the non-zero count does NOT double, it
QUADRUPLES (exactly 4.0x — every one of the four blocks has the scalar sparsity
pattern), and the solve is 2.3x slower.

Two variants of the block run are exercised, because they fail differently:

  * second diffusivity equal to D — v really is a redundant copy, u agrees with
    the scalar answer to 1.1e-16. Pure waste: 2x DOFs, 4x nnz, 2.3x solve time,
    identical physics.
  * second diffusivity carried over from a Schnakenberg template, d_v = 40*D —
    the redundancy breaks, v diffuses 40x faster and leaks into u through the
    coupling: max|u_block - u_scalar| = 3.4e-01 on a field whose true maximum is
    0.527. That is what "spurious cross-diffusion" looks like here, and it is
    the realistic error, since copying the template also copies its d_v/d_u.

Terminology note printed by the fixture: the claim says the "GridFunction"
develops spurious cross-diffusion. GridFunction is NGSolve vocabulary;
``hasattr(skfem, "GridFunction")`` is False on 12.0.1.

Cost: three 40-step runs on a 16x16 grid, about 3 s.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- block_run() drops the second field entirely and performs the scalar
Newton solve on M/dt + D*K - r*M_r that the claim prescribes. The matrix is
then the scalar one, so 'block_matrix_shape=578x578 block_nnz=9604',
'block_ndofs_is_double=True', 'block_nnz_ratio_eq_4=True',
'block_spsolve_slower_gt_1p3x=True' and
'contaminated_block_u_maxdev_gt_0p1=True' disappear from the output.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

import skfem
from skfem import (Basis, BilinearForm, ElementQuad1, LinearForm, MeshQuad,
                   asm)
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"

D_DIFF = 1e-2
R_RATE = 2.0
NX = 16
DT = 0.05
N_STEPS = 40

D2_REDUNDANT = D_DIFF           # v is a true copy of u
D2_WRONG = 40.0 * D_DIFF        # the Schnakenberg d_v/d_u carried over


@LinearForm
def kpp_res(v, w):
    return (w["uu"] * (1.0 - w["uu"])) * v


@BilinearForm
def kpp_jac(u, v, w):
    """M_r: mass matrix weighted by (1 - 2 u^n)."""
    return (1.0 - 2.0 * w["uu"]) * u * v


@LinearForm
def blk_res_u(v, w):
    return (w["uu"] * (1.0 - w["vv"])) * v


@LinearForm
def blk_res_v(v, w):
    return (w["vv"] * (1.0 - w["uu"])) * v


@BilinearForm
def blk_jac_uu(u, v, w):
    return (1.0 - w["vv"]) * u * v


@BilinearForm
def blk_jac_uv(u, v, w):
    return (-w["uu"]) * u * v


@BilinearForm
def blk_jac_vu(u, v, w):
    return (-w["vv"]) * u * v


@BilinearForm
def blk_jac_vv(u, v, w):
    return (1.0 - w["uu"]) * u * v


def setup():
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                                np.linspace(0.0, 1.0, NX + 1))
    basis = Basis(mesh, ElementQuad1())
    xs, ys = basis.doflocs
    u0 = 0.1 * np.exp(-50.0 * ((xs - 0.5) ** 2 + (ys - 0.5) ** 2))
    return basis, laplace.assemble(basis), mass.assemble(basis), u0


def scalar_run():
    basis, K, M, u = setup()
    A0 = M / DT + D_DIFF * K
    t_total, n_solve, shape, nnz = 0.0, 0, None, None
    for _step in range(N_STEPS):
        u_old = u.copy()
        u_new = u_old.copy()
        for _it in range(20):
            U = basis.interpolate(u_new)
            R = (M @ (u_new - u_old) / DT + D_DIFF * K @ u_new
                 - R_RATE * asm(kpp_res, basis, uu=U))
            J = (A0 - R_RATE * asm(kpp_jac, basis, uu=U)).tocsr()
            shape, nnz = J.shape, J.nnz
            t0 = time.perf_counter()
            du = spsolve(J, -R)
            t_total += time.perf_counter() - t0
            n_solve += 1
            u_new += du
            if np.linalg.norm(du) < 1e-11:
                break
        u = u_new
    return basis, u, shape, nnz, t_total / n_solve


def block_run(d2: float):
    """WRONG: the scalar problem set up as a coupled two-field system."""
    if MUTATE:
        # THE PATHOLOGY, removed: Fisher-KPP is scalar, so the second field
        # and its three extra Jacobian blocks are dropped and the claim's
        # single Newton solve on M/dt + D*K - r*M_r is used instead.
        _basis, u, shape, nnz, t_mean = scalar_run()
        return u, shape, nnz, t_mean
    basis, K, M, u = setup()
    v = u.copy()
    N = basis.N
    Au = M / DT + D_DIFF * K
    Av = M / DT + d2 * K
    t_total, n_solve, shape, nnz = 0.0, 0, None, None
    for _step in range(N_STEPS):
        u_old, v_old = u.copy(), v.copy()
        u_new, v_new = u_old.copy(), v_old.copy()
        for _it in range(20):
            U, V = basis.interpolate(u_new), basis.interpolate(v_new)
            R = np.concatenate([
                M @ (u_new - u_old) / DT + D_DIFF * K @ u_new
                - R_RATE * asm(blk_res_u, basis, uu=U, vv=V),
                M @ (v_new - v_old) / DT + d2 * K @ v_new
                - R_RATE * asm(blk_res_v, basis, uu=U, vv=V)])
            J = bmat([[Au - R_RATE * asm(blk_jac_uu, basis, uu=U, vv=V),
                       -R_RATE * asm(blk_jac_uv, basis, uu=U, vv=V)],
                      [-R_RATE * asm(blk_jac_vu, basis, uu=U, vv=V),
                       Av - R_RATE * asm(blk_jac_vv, basis, uu=U, vv=V)]],
                     format="csr")
            shape, nnz = J.shape, J.nnz
            t0 = time.perf_counter()
            dx = spsolve(J, -R)
            t_total += time.perf_counter() - t0
            n_solve += 1
            u_new += dx[:N]
            v_new += dx[N:]
            if np.linalg.norm(dx) < 1e-11:
                break
        u, v = u_new, v_new
    return u, shape, nnz, t_total / n_solve


def main() -> int:
    ok = True
    print(f"element={type(ElementQuad1()).__name__}")
    print(f"skfem_has_GridFunction={hasattr(skfem, 'GridFunction')}")

    # --- RIGHT variant: the scalar system ----------------------------------
    basis, u_s, shape_s, nnz_s, t_s = scalar_run()
    print(f"n_dofs={basis.N} n_elements={basis.mesh.nelements}")
    print(f"scalar_matrix_shape={shape_s[0]}x{shape_s[1]} scalar_nnz={nnz_s}")
    print(f"scalar u=[{u_s.min():.5f}, {u_s.max():.5f}] mean={u_s.mean():.5f}")
    grew = 0.1 < u_s.max() < 1.0
    print(f"scalar_max_u_between_seed_and_carrying_capacity={grew}")
    if not grew:
        print("FAIL: the scalar Fisher-KPP run did not grow toward the carrying "
              "capacity, so there is no healthy reference", file=sys.stderr)
        ok = False

    # --- WRONG variant A: redundant second field, pure waste ---------------
    u_r, shape_b, nnz_b, t_r = block_run(D2_REDUNDANT)
    dev_r = float(np.abs(u_r - u_s).max())
    print(f"block_matrix_shape={shape_b[0]}x{shape_b[1]} block_nnz={nnz_b}")
    print(f"block_ndofs_is_double={shape_b[0] == 2 * shape_s[0]}")
    print(f"block_nnz_over_scalar_nnz={nnz_b / nnz_s:.1f}")
    print(f"block_nnz_ratio_eq_4={abs(nnz_b / nnz_s - 4.0) < 1e-9}")
    print(f"scalar_mean_spsolve_ms={t_s * 1e3:.2f} "
          f"block_mean_spsolve_ms={t_r * 1e3:.2f}")
    print(f"block_solve_time_ratio={t_r / t_s:.2f}")
    print(f"block_spsolve_slower_gt_1p3x={t_r / t_s > 1.3}")
    print(f"redundant_block_u_maxdev={dev_r:.3e}")
    print(f"redundant_block_matches_scalar={dev_r < 1e-12}")
    if not (shape_b[0] == 2 * shape_s[0] and abs(nnz_b / nnz_s - 4.0) < 1e-9):
        print("FAIL: the block system is no longer double the DOFs and four "
              "times the non-zeros", file=sys.stderr)
        ok = False
    if not t_r / t_s > 1.3:
        print("FAIL: the block solve was not measurably more expensive",
              file=sys.stderr)
        ok = False
    if not dev_r < 1e-12:
        print("FAIL: the redundant block run did not reproduce the scalar "
              "answer, so the cost cannot be called pure waste", file=sys.stderr)
        ok = False

    # --- WRONG variant B: the template's second diffusivity comes along ----
    u_c, _shape, _nnz, t_c = block_run(D2_WRONG)
    dev_c = float(np.abs(u_c - u_s).max())
    print(f"contaminated u=[{u_c.min():.5f}, {u_c.max():.5f}]")
    print(f"contaminated_block_u_maxdev={dev_c:.4e}")
    print(f"contaminated_block_u_maxdev_gt_0p1={dev_c > 0.1}")
    print(f"contaminated_relative_to_true_max={dev_c / u_s.max():.2f}")
    if not dev_c > 0.1:
        print("FAIL: carrying the template's second diffusivity did not "
              "contaminate u — the claim's pathology is absent", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
