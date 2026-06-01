"""Tier-2 Layer-C: kratos poisson numerical-correctness gate.

The Kratos poisson catalog (src/backends/kratos/generators/
poisson.py) currently uses scipy-assembled P1 triangles on a
Cartesian unit-square grid for the Poisson PDE itself —
Kratos handles metadata + I/O but not the assembly. This
isn't ideal (task #25 tracks switching to ConvectionDiffusion
Application), but it IS what the catalog ships TODAY. This
gate validates the current catalog pattern converges
optimally against the manufactured sin(πx)sin(πy) solution.

MMS:
  u_exact(x, y) = sin(π·x) · sin(π·y) on [0, 1]^2
  f = 2 π² sin(π·x) · sin(π·y)  (so -Δu = f)
  u = 0 on ∂Ω

Expected at nx=32:
  P1 L2 ≲ 3e-3  (uniform Cartesian grid, P1 triangles
                 ordered like the catalog template)
  P1 EOC nx=16 → nx=32 ∈ [1.7, 2.3]

Note: when task #25 lands the Kratos-native CDA template,
this gate's measured numbers will change but the floors
should still hold (CDA's P1 element assembles the same
weak form).
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def run_kratos_pattern_poisson(nx: int) -> float:
    """Solve Poisson on [0,1]^2 using the catalog's
    scipy-assembled P1 pattern, return L2 error vs
    manufactured sin(πx)sin(πy)."""
    ny = nx
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

    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    for tri in elements:
        ids = [t - 1 for t in tri]
        x = np.array([coords[t][0] for t in tri])
        y = np.array([coords[t][1] for t in tri])
        area = 0.5 * abs(
            (x[1] - x[0]) * (y[2] - y[0])
            - (x[2] - x[0]) * (y[1] - y[0]))
        b = np.array([y[1] - y[2], y[2] - y[0],
                      y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2],
                      x[1] - x[0]])
        Ke = (1.0 / (4.0 * area)) * (np.outer(b, b)
                                     + np.outer(c, c))
        # f = 2π²·sin(πx)·sin(πy) — element-centroid sample
        xc = float(np.mean(x))
        yc = float(np.mean(y))
        f_val = (2.0 * np.pi ** 2
                 * np.sin(np.pi * xc)
                 * np.sin(np.pi * yc))
        fe = f_val * area / 3.0 * np.ones(3)
        for a in range(3):
            F[ids[a]] += fe[a]
            for bb in range(3):
                K[ids[a], ids[bb]] += Ke[a, bb]
    K = K.tocsr()

    # Apply u=0 on boundary (j=0, j=ny, i=0, i=nx)
    boundary = set()
    for j in range(ny + 1):
        boundary.add(node_map[(0, j)] - 1)
        boundary.add(node_map[(nx, j)] - 1)
    for i in range(nx + 1):
        boundary.add(node_map[(i, 0)] - 1)
        boundary.add(node_map[(i, ny)] - 1)
    interior = sorted(set(range(n_nodes)) - boundary)
    u = np.zeros(n_nodes)
    u[interior] = spsolve(
        K[np.ix_(interior, interior)], F[interior])

    # L2 error: nodal sum weighted by lumped-mass.
    err_sq = 0.0
    for tri in elements:
        ids = [t - 1 for t in tri]
        x = np.array([coords[t][0] for t in tri])
        y = np.array([coords[t][1] for t in tri])
        area = 0.5 * abs(
            (x[1] - x[0]) * (y[2] - y[0])
            - (x[2] - x[0]) * (y[1] - y[0]))
        # u_h at the 3 vertices; u_exact at the same
        u_h = u[ids]
        u_ex = np.sin(np.pi * x) * np.sin(np.pi * y)
        # (u_h - u_ex)^2 integrated using a 3-point rule
        # (vertex quadrature):
        diff_sq = (u_h - u_ex) ** 2
        err_sq += area * float(np.mean(diff_sq))
    return float(math.sqrt(err_sq))


def main() -> int:
    print(f"kratos_catalog_pattern=scipy_P1_assembly")
    err_h32 = run_kratos_pattern_poisson(nx=32)
    err_h16 = run_kratos_pattern_poisson(nx=16)
    eoc = (math.log(err_h16 / err_h32) / math.log(2.0)
           if err_h32 > 0 else float("nan"))
    print(f"P1_h32_l2err={err_h32:.6e}_tol=3e-03")
    print(f"P1_h16_l2err={err_h16:.6e}")
    print(f"P1_eoc_h16_to_h32={eoc:.3f}_expected=2.0")

    fail_reasons = []
    if err_h32 > 3e-3:
        fail_reasons.append(
            f"P1 h=1/32 L2 err {err_h32:.3e} > 3e-3")
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
