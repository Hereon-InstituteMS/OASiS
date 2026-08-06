"""Tier-2: the shipped heat_transient/2d time loop is not backward Euler.

EXERCISES THE NUMPY/SCIPY PATH, NOT KRATOS. The catalog template
src/backends/kratos/generators/heat.py::_heat_transient_2d_kratos
says "Kratos (manual assembly)" on its first line and assembles
K and M with numpy/scipy; Kratos is not called. This fixture
replays that template's own time loop verbatim and judges it on
physics.

Pitfall (kratos.heat_transient #0) is "Backward Euler: factor
(M + dt*K) once and reuse each step". The template does factor
once — and then does not solve backward Euler.

What the loop computes, algebraically. Per step it forms

    rhs  = M @ T
    rhs -= A @ T                 # A = M + dt*K
    rhs[dirichlet] = 0.0         # does not touch interior rows
    T_new[interior] = A_ii^-1 (rhs + M @ T)[interior]

so the interior right-hand side is (2M - A) @ T = (M - dt*K) @ T,
i.e. it solves (M + dt*K) T_new = (M - dt*K) T_old. That is not
backward Euler, and the Dirichlet columns are carried on the OLD
step only — A_ib T_b^new is never subtracted. As dt -> 0 the
per-step increment tends to M_ii^-1 M_ib T_b, a constant that
does NOT vanish with dt, so the error grows without bound as the
step is refined.

THE OBSERVABLE: for pure diffusion with no source and Dirichlet
data in [0, 100], the discrete maximum principle forbids
max(T) > 100. Refining dt at fixed end time drives the shipped
loop far above it, while a correct backward Euler stays at
exactly 100.

No measured number from this run is written into any knowledge
text; the assertions below are this fixture's own output.

Mutation control: T2_MUTATE=1 flips SHIPPED_LOOP to False, so the corrected backward-Euler loop is exercised instead of the shipped one. The maximum-principle violation is exactly what the shipped loop causes, so it disappears: loop_under_test reads corrected, no refinement level exceeds the 100 K boundary maximum (which also trips the forbidden string steps_above_boundary_max=0) and max(T) stops growing under dt refinement.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=shipped_loop_replaced_by_the_corrected_backward_euler_loop")

# Set False by the mutation gate to run the corrected loop instead.
SHIPPED_LOOP = not MUTATE

NX = 16
T_LEFT, T_RIGHT = 100.0, 0.0
T_END = 0.1


def build(nx: int, kappa: float = 1.0, rho_cp: float = 1.0):
    """K, M and the grid, assembled exactly as the template does."""
    ny = nx
    nid = 1
    node_map: dict[tuple[int, int], int] = {}
    coords: dict[int, tuple[float, float]] = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            coords[nid] = (i / nx, j / ny)
            node_map[(i, j)] = nid
            nid += 1
    n = nid - 1

    elements = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (node_map[(i, j)], node_map[(i + 1, j)],
                          node_map[(i + 1, j + 1)], node_map[(i, j + 1)])
            elements.append((a, b, d))
            elements.append((b, c, d))

    K = lil_matrix((n, n))
    M = lil_matrix((n, n))
    for tri in elements:
        ids = [t - 1 for t in tri]
        x = np.array([coords[t][0] for t in tri])
        y = np.array([coords[t][1] for t in tri])
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0])
                         - (x[2] - x[0]) * (y[1] - y[0]))
        bb = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        cc = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        Ke = kappa * (1.0 / (4.0 * area)) * (np.outer(bb, bb) + np.outer(cc, cc))
        Me = rho_cp * area / 12.0 * (np.ones((3, 3)) + np.eye(3))
        for p in range(3):
            for q in range(3):
                K[ids[p], ids[q]] += Ke[p, q]
                M[ids[p], ids[q]] += Me[p, q]
    return K.tocsr(), M.tocsr(), node_map, n, nx, ny


K, M, NODE_MAP, N, NX, NY = build(NX)
LEFT = {NODE_MAP[(0, j)] - 1 for j in range(NY + 1)}
RIGHT = {NODE_MAP[(NX, j)] - 1 for j in range(NY + 1)}
DIRICH = LEFT | RIGHT
INTERIOR = sorted(set(range(N)) - DIRICH)
DB_IDX = sorted(DIRICH)


def shipped_run(dt: float, nsteps: int) -> np.ndarray:
    """The template's loop, copied line for line from heat.py."""
    A = M + dt * K
    solve_A = factorized(A[np.ix_(INTERIOR, INTERIOR)].tocsc())
    T = np.zeros(N)
    for q in LEFT:
        T[q] = T_LEFT
    for q in RIGHT:
        T[q] = T_RIGHT
    for _ in range(nsteps):
        rhs = M @ T
        rhs -= A @ T
        rhs[list(DIRICH)] = 0.0
        T_new = T.copy()
        T_new[INTERIOR] = solve_A(rhs[INTERIOR] + (M @ T)[INTERIOR])
        for q in LEFT:
            T_new[q] = T_LEFT
        for q in RIGHT:
            T_new[q] = T_RIGHT
        T = T_new
    return T


def backward_euler(dt: float, nsteps: int) -> np.ndarray:
    """(M + dt K) T_new = M T_old with the Dirichlet columns eliminated
    on the NEW step, which is what the template's docstring promises."""
    A = M + dt * K
    Aii = A[np.ix_(INTERIOR, INTERIOR)].tocsc()
    Aib = A[np.ix_(INTERIOR, DB_IDX)]
    Mii = M[np.ix_(INTERIOR, INTERIOR)]
    Mib = M[np.ix_(INTERIOR, DB_IDX)]
    solve = factorized(Aii)
    T = np.zeros(N)
    for q in LEFT:
        T[q] = T_LEFT
    for q in RIGHT:
        T[q] = T_RIGHT
    db = np.array([T[q] for q in DB_IDX])
    for _ in range(nsteps):
        T[INTERIOR] = solve(Mii @ T[INTERIOR] + Mib @ db - Aib @ db)
    return T


def main() -> int:
    bad = 0
    loop = shipped_run if SHIPPED_LOOP else backward_euler
    print(f"exercises=numpy_scipy_template_path_not_kratos")
    print(f"loop_under_test={'shipped' if SHIPPED_LOOP else 'corrected'}")

    maxima = []
    for dt in (1e-3, 5e-4, 2.5e-4, 1e-4):
        nsteps = int(round(T_END / dt))
        T = loop(dt, nsteps)
        ref = backward_euler(dt, nsteps)
        maxima.append(T.max())
        print(f"dt={dt:.2e}_steps={nsteps}_maxT={T.max():.4f}"
              f"_refmaxT={ref.max():.4f}")
        if ref.max() > 100.0 + 1e-9:
            print(f"FAIL: the backward-Euler reference itself broke the "
                  f"maximum principle at dt={dt}", file=sys.stderr)
            bad += 1

    # 1. The shipped loop must exceed the boundary maximum at some dt.
    over = [m for m in maxima if m > 100.0 + 1e-9]
    print(f"steps_above_boundary_max={len(over)}")
    if not over:
        print("FAIL: no refinement level exceeded the 100 K boundary "
              "maximum — the maximum-principle violation did not reproduce",
              file=sys.stderr)
        bad += 1

    # 2. Refining dt must make it WORSE, not better. A consistent scheme
    #    converges; this one diverges.
    print(f"maxT_grows_under_refinement={maxima[-1] > maxima[0] + 1.0}")
    if not maxima[-1] > maxima[0] + 1.0:
        print(f"FAIL: max(T) did not grow under dt refinement "
              f"({maxima[0]:.4f} -> {maxima[-1]:.4f}); the scheme would be "
              f"consistent", file=sys.stderr)
        bad += 1

    print(f"heat_transient_maximum_principle_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
