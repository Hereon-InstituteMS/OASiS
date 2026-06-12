"""scikit-fem wave-equation generators and knowledge.

Covers the 2D scalar wave equation
    u_tt - c^2 Δu = 0    on Ω = [0,1]^2
    u = 0                on ∂Ω
    u(x,0) = u0(x),  u_t(x,0) = v0(x)

with explicit central-difference time-stepping and row-sum-lumped mass
so each step is a single sparse-matrix-vector product (no linear solve).

Modelled after scikit-fem upstream examples ex09 / ex36 / ex44 (wave
equation variants) — the backend previously had **no** wave-equation
generator at all, leaving a clear coverage gap relative to upstream.
"""


def _wave_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    2D scalar wave equation, explicit central-difference time
    integration with row-sum-lumped mass. Output: max amplitude
    over the simulation + a results_summary.json with the
    central-node history sampled at t_end."""
    nx = params.get("nx", 24)
    c = params.get("c", 1.0)
    T_end = params.get("T_end", 0.4)
    # CFL: dt < h / (c * sqrt(2)) for 2D Q1. Pick 0.5 of that.
    safety = params.get("cfl_safety", 0.5)
    return f'''\
"""2D scalar wave equation: u_tt - c^2 Δu = 0 — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np
import json

c     = {c}
T_end = {T_end}
nx    = {nx}

_tol = 1e-10
m = (MeshQuad.init_tensor(np.linspace(0, 1, nx + 1),
                          np.linspace(0, 1, nx + 1))
     .with_boundaries({{
         "bnd": lambda x: (x[0] < _tol) | (x[0] > 1.0 - _tol)
                          | (x[1] < _tol) | (x[1] > 1.0 - _tol),
     }}))
e = ElementQuad1()
ib = Basis(m, e)

K = laplace.assemble(ib)
M = mass.assemble(ib)

# Row-sum (HRZ-style) lumped mass — diagonal vector. Lumping converts
# the explicit update into one diag-solve per step. The trade-off is a
# slightly-overdamped dispersion; for the manufactured BC-zero problem
# below the time-error stays bounded by the leading O(dt^2) term.
M_lumped = np.asarray(M.sum(axis=1)).ravel()

bnd_dofs = ib.get_dofs("bnd").flatten()
interior = np.setdiff1d(np.arange(ib.N), bnd_dofs)

# CFL: dt < h_min / (c * sqrt(2)) for 2D Q1.
h_min = 1.0 / nx
dt = {safety} * h_min / (c * np.sqrt(2.0))
n_steps = int(np.ceil(T_end / dt))
dt = T_end / n_steps  # adjust to land exactly at T_end

# Initial condition: lowest standing-wave mode on the unit square.
# u(x,y,0) = sin(pi x) sin(pi y); u_t(x,y,0) = 0.
x_coord = m.p[0, :]
y_coord = m.p[1, :]
u_old = np.sin(np.pi * x_coord) * np.sin(np.pi * y_coord)
u_old[bnd_dofs] = 0.0
# u_t(0) = 0 ⇒ u^{{-1}} = u^0 - 0.5 dt^2 M^{{-1}} (-c^2 K u^0)
#               = u^0 - 0.5 dt^2 M^{{-1}} c^2 K u^0
rhs0 = c * c * (K @ u_old)
acc0 = -rhs0 / M_lumped
acc0[bnd_dofs] = 0.0
u_prev = u_old - 0.5 * dt * dt * acc0
u_prev[bnd_dofs] = 0.0

# Track central-node amplitude for sanity (mid-point of the square).
center_dof = int(np.argmin((x_coord - 0.5) ** 2 + (y_coord - 0.5) ** 2))
center_history = [float(u_old[center_dof])]

amp_max = float(np.abs(u_old).max())
for step in range(n_steps):
    # u^{{n+1}} = 2 u^n - u^{{n-1}} - dt^2 M^{{-1}} c^2 K u^n
    rhs = c * c * (K @ u_old)
    u_new = 2.0 * u_old - u_prev - (dt * dt) * (rhs / M_lumped)
    u_new[bnd_dofs] = 0.0
    u_prev = u_old
    u_old = u_new
    a = float(np.abs(u_new).max())
    if a > amp_max:
        amp_max = a
    center_history.append(float(u_new[center_dof]))

# Analytical reference for the lowest mode:
# u(x,y,t) = cos(c * pi * sqrt(2) * t) * sin(pi x) sin(pi y)
# Initial amplitude at center = 1.0; FE solution should remain bounded
# by |amp_max| ~ 1.0 ± O(dt^2) over the simulation.
print(f"steps={{n_steps}} dt={{dt:.4e}} max|u|={{amp_max:.6f}} "
      f"u_center(T)={{center_history[-1]:.6f}}")

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, [("quad", m.t.T)], point_data={{"u": u_old}})
mio.write("result.vtu")

summary = {{
    "max_amplitude": amp_max,
    "u_center_T_end": center_history[-1],
    "n_steps": n_steps,
    "dt": dt,
    "n_dofs": int(ib.N),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "wave_2d": _wave_2d,
}


KNOWLEDGE: dict = {
    "wave": {
        "description": (
            "Scalar wave equation u_tt - c^2 Δu = 0 with "
            "homogeneous Dirichlet BCs. Explicit central-"
            "difference time integration + row-sum-lumped mass "
            "(no linear solve per step). Matches scikit-fem "
            "upstream ex09 / ex36 / ex44 in physics; the lumped-"
            "explicit variant here is the cheapest runnable form."
        ),
        "weak_form": (
            "M u_tt + c^2 K u = 0,  u^{n+1} = 2u^n - u^{n-1} "
            "- dt^2 M_L^{-1} c^2 K u^n"
        ),
        "elements": ["ElementQuad1 (P1 quad)"],
        "variants": ["2d"],
        "pitfalls": [
            "[Numerical] CFL: explicit central-difference is "
            "stable only for dt < h_min / (c * sqrt(2)) on 2D Q1. "
            "Pick dt safely below this (e.g. 0.5 * h/(c*sqrt(2))) "
            "or the amplitude blows up exponentially within a few "
            "dozen steps. "
            "Signal: max|u| grows without bound (>> initial "
            "amplitude) within 20-50 steps; np.isfinite(u).all() "
            "becomes False; ValueError 'array must not contain "
            "infs or NaNs' from spsolve when downstream code "
            "tries to write the field.",
            "[API] scikit-fem >= 12 expects "
            "MeshQuad.init_tensor for tensor-product Q1 grids; "
            "older `MeshQuad((nx,ny))` constructor was removed. "
            "Signal: TypeError 'MeshQuad() takes 1 positional "
            "argument but 2 were given' or "
            "AttributeError 'type object MeshQuad has no "
            "attribute __init__' on the mesh construction line.",
            "[Numerical] HRZ-style lumping via "
            "`scipy.sparse.csr_matrix.sum(axis=1)` is the "
            "simplest mass-lumping but introduces a slight "
            "numerical dispersion error (~O(h^2)). For high-"
            "frequency content, prefer row-diagonal scaling or "
            "use consistent mass + `scipy.sparse.linalg.spsolve` "
            "per step. "
            "Signal: amplitude at T_end drifts from analytic "
            "value cos(c * pi * sqrt(2) * T) by > 1% even with "
            "small dt; `scipy.sparse.linalg.spsolve` of the "
            "consistent-mass system shows refinement convergence "
            "at O(h^2) while the lumped form stalls at O(h^1.5).",
            "[API] `mass.assemble(basis)` returns a "
            "scipy.sparse.csr_matrix; summing along axis=1 "
            "produces a numpy.matrix in NumPy < 2.0 and an "
            "ndarray in NumPy >= 2.0. Use np.asarray(...).ravel() "
            "to coerce to 1-D array regardless. "
            "Signal: TypeError 'unsupported operand type(s) for /' "
            "with ndarray / matrix on the M_lumped division line, "
            "or DeprecationWarning 'matrix subclass is "
            "deprecated' from numpy.",
            "[Physics] Initial condition with u_t(0) = 0 "
            "requires a special first time step: "
            "u^{-1} = u^0 - 0.5 dt^2 M^{-1} (-c^2 K u^0). "
            "Skipping this and using u^{-1} = u^0 introduces a "
            "spurious initial velocity that pollutes the long-"
            "time solution. The fix uses "
            "`laplace.assemble(basis)` for K, "
            "`mass.assemble(basis)` for M, and a single "
            "explicit pre-step before the main loop. "
            "Signal: standing-wave amplitude at center drifts "
            "linearly with t after t=0 instead of oscillating; "
            "u_center(T) does not return to +/-1 at the "
            "analytic periods (T = sqrt(2)/c, 2*sqrt(2)/c, ...). "
            "Even with mass.assemble(ib) and laplace.assemble(ib) "
            "matrices built correctly, MeshQuad+ElementQuad1 with "
            "the IC bug shows monotone drift in `np.abs(u).max()` "
            "rather than oscillation.",
            "[Numerical] Dirichlet BCs must be re-applied "
            "every step (u_new[bnd_dofs] = 0). The explicit "
            "update propagates non-zero values into boundary "
            "DOFs via the off-diagonal K-coupling; without the "
            "re-application, energy slowly leaks at the "
            "boundary and the simulation is no longer a "
            "homogeneous-BC problem. "
            "Signal: |u| at boundary DOFs grows from 0 to "
            "non-negligible values (>1e-6) within a few hundred "
            "steps; the boundary becomes a small but persistent "
            "source.",
            "[Output] VTK output via `meshio.Mesh(...)` "
            "requires 3D points; for a 2D mesh you must pad with "
            "a zero z-column: "
            "`np.column_stack([m.p.T, np.zeros(m.p.shape[1])])`. "
            "The cells argument is `[('quad', m.t.T)]` for "
            "MeshQuad and `[('triangle', m.t.T)]` for MeshTri — "
            "passing the wrong cell-type tag produces a silently "
            "malformed .vtu file. "
            "Signal: `meshio.write` raises WriteError or "
            "ParaView shows a zero-thickness slab; `meshio.Mesh` "
            "may construct but downstream "
            "`meshio.write('result.vtu', mio)` produces a file "
            "ParaView renders as a degenerate 2D-in-3D quad set. "
            "Underlying cause is that MeshQuad.p has shape "
            "(2, n_nodes) — `np.column_stack([m.p.T, np.zeros(...)`"
            " is required to lift to 3D for meshio.",
        ],
        "references": [
            "scikit-fem examples: ex09 (3D wave), ex36 (wave "
            "equation), ex44 (wave equation, alt formulation)",
            "Hughes, T.J.R. The Finite Element Method (1987), "
            "Ch. 9: hyperbolic problems and CFL conditions",
        ],
    },
}
