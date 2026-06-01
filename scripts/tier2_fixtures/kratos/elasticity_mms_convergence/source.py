"""Tier-2 Layer-C: kratos linear_elasticity MMS gate.

The Kratos elasticity catalog (src/backends/kratos/
generators/linear_elasticity.py) uses a scipy-Voigt-B-
matrix manual assembly pattern, mirroring the Poisson
catalog approach (task #25 tracks the upgrade to a
KratosMultiphysics StructuralMechanicsApplication
template — floors set here will still hold when that
lands). Plane-strain D-matrix in engineering-strain
Voigt notation:

  D = [[λ+2μ, λ,    0],
       [λ,    λ+2μ, 0],
       [0,    0,    μ]]      (μ — engineering γ_xy)

Cross-backend mirror of fenics + skfem + ngsolve
elasticity_mms_convergence. Same MMS:

  u_exact = (sin(π·x)·sin(π·y),
             sin(2π·x)·sin(2π·y))
  plane strain, E = 1.0, ν = 0.3
  f = -μ·Δu - (λ + μ)·grad(div(u))     (Navier form)

Distinct API: scipy lil_matrix, Voigt B-matrix
assembly, interleaved x/y dofs, spsolve direct.

Expected at nx=32 (uniform Cartesian):
  P1 L2 ≲ 1.3e-2  (similar to skfem 8.96e-3 + margin
                   for centroid-quadrature error)
  P1 EOC nx=16 → nx=32 ∈ [1.7, 2.3]
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def run_elasticity_mms(nx: int) -> float:
    """Solve elasticity MMS via the catalog's scipy-
    Voigt-B-matrix pattern. Return L2 error vs the
    manufactured solution."""
    ny = nx
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    nid = 1
    node_map: dict[tuple[int, int], int] = {}
    coords: dict[int, tuple[float, float]] = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            coords[nid] = (i / nx, j / ny)
            node_map[(i, j)] = nid
            nid += 1
    n_nodes = nid - 1

    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i + 1, j)]
            n3 = node_map[(i + 1, j + 1)]
            n4 = node_map[(i, j + 1)]
            elements.append((n1, n2, n4))
            elements.append((n2, n3, n4))

    ndof = 2 * n_nodes
    K = lil_matrix((ndof, ndof))
    F = np.zeros(ndof)
    D_mat = np.array([
        [lam + 2.0 * mu, lam, 0.0],
        [lam, lam + 2.0 * mu, 0.0],
        [0.0, 0.0, mu],
    ])

    def f_at(xc: float, yc: float) -> tuple[float, float]:
        """f = -μ·Δu - (λ + μ)·grad(div(u))
        evaluated at point (xc, yc)."""
        pi = math.pi
        sin_px = math.sin(pi * xc)
        sin_py = math.sin(pi * yc)
        cos_px = math.cos(pi * xc)
        cos_py = math.cos(pi * yc)
        sin_2px = math.sin(2.0 * pi * xc)
        sin_2py = math.sin(2.0 * pi * yc)
        cos_2px = math.cos(2.0 * pi * xc)
        cos_2py = math.cos(2.0 * pi * yc)
        lap_u1 = -2.0 * pi ** 2 * sin_px * sin_py
        lap_u2 = -8.0 * pi ** 2 * sin_2px * sin_2py
        gdiv_x = (-(pi ** 2) * sin_px * sin_py
                  + 4.0 * (pi ** 2) * cos_2px * cos_2py)
        gdiv_y = ((pi ** 2) * cos_px * cos_py
                  - 4.0 * (pi ** 2) * sin_2px * sin_2py)
        f1 = -mu * lap_u1 - (lam + mu) * gdiv_x
        f2 = -mu * lap_u2 - (lam + mu) * gdiv_y
        return f1, f2

    for tri in elements:
        ids = [t - 1 for t in tri]
        x = np.array([coords[t][0] for t in tri])
        y = np.array([coords[t][1] for t in tri])
        area = 0.5 * abs(
            (x[1] - x[0]) * (y[2] - y[0])
            - (x[2] - x[0]) * (y[1] - y[0]))
        b = np.array([y[1] - y[2], y[2] - y[0],
                      y[0] - y[1]]) / (2.0 * area)
        c = np.array([x[2] - x[1], x[0] - x[2],
                      x[1] - x[0]]) / (2.0 * area)
        B = np.zeros((3, 6))
        for a in range(3):
            B[0, 2 * a] = b[a]
            B[1, 2 * a + 1] = c[a]
            B[2, 2 * a] = c[a]
            B[2, 2 * a + 1] = b[a]
        Ke = area * B.T @ D_mat @ B
        # Element body-force vector: f at the centroid
        # times area / 3 distributed to each vertex.
        xc = float(np.mean(x))
        yc = float(np.mean(y))
        f1, f2 = f_at(xc, yc)
        Fe = np.zeros(6)
        for a in range(3):
            Fe[2 * a] = f1 * area / 3.0
            Fe[2 * a + 1] = f2 * area / 3.0
        dofs = []
        for a in range(3):
            dofs.extend([2 * ids[a], 2 * ids[a] + 1])
        for ii in range(6):
            F[dofs[ii]] += Fe[ii]
            for jj in range(6):
                K[dofs[ii], dofs[jj]] += Ke[ii, jj]
    K = K.tocsr()

    # Dirichlet u = u_exact on ALL boundaries.
    fixed_indices = []
    fixed_values = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            n = node_map[(i, j)] - 1
            xc, yc = coords[n + 1]
            if (i == 0 or i == nx or j == 0
                    or j == ny):
                u_x = math.sin(math.pi * xc) * math.sin(
                    math.pi * yc)
                u_y = math.sin(2.0 * math.pi * xc) * math.sin(
                    2.0 * math.pi * yc)
                fixed_indices.append(2 * n)
                fixed_values.append(u_x)
                fixed_indices.append(2 * n + 1)
                fixed_values.append(u_y)
    fixed_set = set(fixed_indices)
    interior = sorted(set(range(ndof)) - fixed_set)

    u = np.zeros(ndof)
    for idx, val in zip(fixed_indices, fixed_values):
        u[idx] = val
    # Move BC contribution to RHS: F_int - K_int_bc · u_bc
    K_csr = K
    rhs = F.copy()
    rhs -= K_csr @ u
    u[interior] = spsolve(
        K_csr[np.ix_(interior, interior)],
        rhs[interior])

    # L2 error via vertex quadrature.
    err_sq = 0.0
    for tri in elements:
        ids = [t - 1 for t in tri]
        x = np.array([coords[t][0] for t in tri])
        y = np.array([coords[t][1] for t in tri])
        area = 0.5 * abs(
            (x[1] - x[0]) * (y[2] - y[0])
            - (x[2] - x[0]) * (y[1] - y[0]))
        u_h_x = np.array([u[2 * idx] for idx in ids])
        u_h_y = np.array([u[2 * idx + 1] for idx in ids])
        u_ex_x = np.sin(np.pi * x) * np.sin(np.pi * y)
        u_ex_y = (np.sin(2.0 * np.pi * x)
                  * np.sin(2.0 * np.pi * y))
        diff_sq = ((u_h_x - u_ex_x) ** 2
                   + (u_h_y - u_ex_y) ** 2)
        err_sq += area * float(np.mean(diff_sq))
    return float(math.sqrt(err_sq))


def main() -> int:
    print(f"kratos_catalog_pattern=scipy_voigt_P1_assembly")
    err_h32 = run_elasticity_mms(nx=32)
    err_h16 = run_elasticity_mms(nx=16)
    eoc = (math.log(err_h16 / err_h32) / math.log(2.0)
           if err_h32 > 0 else float("nan"))
    print(f"P1_h32_l2err={err_h32:.6e}_tol=1.3e-02")
    print(f"P1_h16_l2err={err_h16:.6e}")
    print(f"P1_eoc_h16_to_h32={eoc:.3f}_expected=2.0")

    fail_reasons = []
    if err_h32 > 1.3e-2:
        fail_reasons.append(
            f"P1 h=1/32 L2 err {err_h32:.3e} > 1.3e-2")
    if not (1.7 <= eoc <= 2.3):
        fail_reasons.append(
            f"P1 EOC {eoc:.3f} outside [1.7, 2.3]")

    if not fail_reasons:
        return 0
    for r in fail_reasons:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
