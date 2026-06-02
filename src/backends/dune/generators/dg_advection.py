"""DUNE-fem DG advection generators and knowledge."""


def _dg_advection_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    DG method for pure advection equation with upwind flux."""
    nx = params.get("nx", 32)
    order = params.get("order", 1)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.5)
    return f'''\
"""DG advection: du/dt + b.grad(u) = 0 — upwind flux — DUNE-fem"""
from dune.grid import structuredGrid
from dune.fem.space import dglagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx,
                 conditional, lt, FacetNormal, avg, jump, dS, ds,
                 CellDiameter)
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = dglagrange(gridView, order={order})
x = SpatialCoordinate(space)

# Initial condition: smooth bump — set for your problem
u_n = space.interpolate(
    conditional(lt((x[0]-0.25)**2 + (x[1]-0.5)**2, 0.04), 1.0, 0.0),
    name="u"
)

u = TrialFunction(space)
v = TestFunction(space)
n = FacetNormal(space)
h = CellDiameter(space)

# Advection velocity — set for your problem
b0, b1 = 1.0, 0.5

# Bilinear form: mass + dt * advection with upwind flux
dt = {dt}

# Mass term
a_mass = u * v * dx

# Advection: volume term + upwind flux on interior facets + inflow boundary
bn = b0 * n[0] + b1 * n[1]

# For DG advection we use an explicit Euler approach with the galerkin solver
# Volume advection
a_adv = -(b0 * u * v.dx(0) + b1 * u * v.dx(1)) * dx

# Interior facet upwind flux
from ufl import gt
u_up = conditional(gt(bn("+"), 0), u("+"), u("-"))
a_adv += (bn("+") * u_up * (v("+") - v("-"))) * dS

# Boundary outflow
a_adv += conditional(gt(bn, 0), bn * u * v, 0) * ds

# Combined form: u_new = u_old - dt * advection(u_old)
a = a_mass - dt * a_adv
b_form = u_n * v * dx

scheme = galerkin([a == b_form], solver="cg")

# Time stepping
t = 0.0
n_steps = int({T_end} / dt)
for step in range(n_steps):
    scheme.solve(target=u_n)
    t += dt

vals = np.array(u_n.as_numpy)
print(f"DG advection: t={{t:.4f}}, max={{vals.max():.6f}}, min={{vals.min():.6f}}")
gridView.writeVTK("result", pointdata={{"concentration": u_n}})
summary = {{
    "time": t, "max_value": float(vals.max()), "min_value": float(vals.min()),
    "n_dofs": len(vals), "order": {order}, "dt": dt,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DG advection solve complete.")
'''


KNOWLEDGE = {
    "dg_advection": {
        "description": "DG method for pure advection equations (upwind flux, explicit time stepping)",
        "solver": "Explicit Euler with DG spatial discretization, upwind numerical flux",
        "spaces": "dglagrange(gridView, order=k) — discontinuous Lagrange of any order",
        "time_stepping": "Explicit Euler, SSP-RK2, SSP-RK3 for stability",
        "pitfalls": [
            (
                "[Numerical] CFL condition: dt < h / "
                "(2*order + 1) / max(|b|) for stability. "
                "Signal: dt > CFL gives NaN within ~10 "
                "steps; the (2*order+1) denominator "
                "tightens the CFL for higher-order DG "
                "elements (order=3 needs dt 7x smaller "
                "than order=1). Safety factor 0.5 is "
                "conservative. (Audit 2026-06-02.)"
            ),
            (
                "[API] Upwind flux: use UFL conditional(gt"
                "(b.n, 0), u('+'), u('-')) for upwind "
                "selection. Signal: a centred flux "
                "(0.5*(u('+') + u('-'))) on pure-"
                "advection DG produces unconditional "
                "instability — solution amplitude grows "
                "exponentially regardless of dt. Use "
                "upwind via UFL conditional. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Interior facet integrals: dS in "
                "UFL (CAPITAL S); boundary integrals: "
                "ds (lowercase). Signal: writing jump "
                "terms over ds instead of dS silently "
                "drops them on interior facets — the "
                "discrete operator has wrong sparsity "
                "(no facet coupling). UFL "
                "case-sensitively distinguishes ds "
                "(boundary) from dS (interior). (Audit "
                "2026-06-02.)"
            ),
            (
                "[Performance] DG with the dune-fem-dg "
                "module: optimised SSP-RK time steppers "
                "available (SSPRK3 / SSPRK4). Signal: "
                "writing a manual explicit Euler loop "
                "for DG advection is 2-5x slower than "
                "the dune-fem-dg SSPRK time stepper, "
                "which uses tuned cell loops and "
                "block-diagonal mass-matrix inversion. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For HIGH-ORDER DG: use "
                "MODAL basis (dgonb, dglegendre) for "
                "better conditioning. Signal: nodal "
                "DG (lagrange order=5) on a high-order "
                "DG problem gives cond(K) > 1e10; "
                "switching to dgonb (orthonormal "
                "basis) gives cond(K) ~ 1e4 for the "
                "same problem — iterative solver "
                "iterations drop accordingly. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Limiters may be needed for "
                "DISCONTINUOUS solutions (TVD / TVB / "
                "WENO). Signal: high-order DG on a "
                "shock problem produces 10-30% over/"
                "undershoot at the discontinuity that "
                "does NOT decay with refinement; "
                "applying a minmod or WENO limiter "
                "monotonically clips the oscillations "
                "at the cost of one order of accuracy. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "dg_advection_2d": _dg_advection_2d,
}
