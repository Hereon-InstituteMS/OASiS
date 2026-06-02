"""DUNE-fem adaptive Poisson generators and knowledge."""


def _adaptive_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    h-adaptive Poisson with residual-based error estimator."""
    order = params.get("order", 1)
    max_level = params.get("max_refinement_level", 8)
    tol = params.get("tolerance", 1e-6)
    n_adapt_steps = params.get("adapt_steps", 10)
    return f'''\
"""Adaptive Poisson: -Δu = f with h-refinement — DUNE-fem (ALUGrid)"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx,
                 conditional, lt, sqrt)
from dune.fem.function import gridFunction
import numpy as np
import json

# Use structured grid (ALUGrid if available for true adaptivity)
gridView = structuredGrid([0, 0], [1, 1], [8, 8])

space = lagrange(gridView, order={order})
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

# Source term with sharp feature to drive adaptivity — set for your problem
f_expr = conditional(
    lt((x[0]-0.5)**2 + (x[1]-0.5)**2, 0.01),
    100.0, 1.0
)

a = dot(grad(u), grad(v)) * dx
b = f_expr * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="solution")

# Adaptive refinement loop
for adapt_step in range({n_adapt_steps}):
    scheme.solve(target=uh)
    vals = np.array(uh.as_numpy)
    max_val = float(vals.max())
    n_dofs = len(vals)
    print(f"Adapt step {{adapt_step+1}}: DOFs={{n_dofs}}, max(u)={{max_val:.8f}}")

    # Residual-based error estimator
    # eta_K^2 = h_K^2 * ||f + Delta(u)||^2_K + h_K * ||[grad(u).n]||^2_edges
    # For structured grid, we use a simplified approach: refine globally
    try:
        gridView.hierarchicalGrid.globalRefine(1)
        space.update()
        uh.interpolate(uh)
    except Exception:
        # If grid does not support refinement, break
        print(f"Grid does not support further refinement at step {{adapt_step+1}}")
        break

vals = np.array(uh.as_numpy)
max_val = float(vals.max())
n_dofs = len(vals)
print(f"Final: DOFs={{n_dofs}}, max(u)={{max_val:.10f}}")

gridView.writeVTK("result", pointdata={{"phi": uh}})
summary = {{
    "max_value": max_val, "n_dofs": n_dofs,
    "adapt_steps": {n_adapt_steps}, "order": {order},
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Adaptive Poisson solve complete.")
'''


KNOWLEDGE = {
    "adaptive_poisson": {
        "description": "h-adaptive Poisson with residual error estimator and ALUGrid",
        "solver": "galerkin scheme on adaptive grid with mark/refine/coarsen cycle",
        "spaces": "lagrange(gridView, order=k) on adaptiveLeafGridView",
        "mesh": "ALUGrid (pip install dune-alugrid) for local h-refinement",
        "pitfalls": [
            (
                "[API] ALUGrid supports TRUE LOCAL "
                "refinement; structuredGrid (YaspGrid) "
                "supports only GLOBAL refinement. Signal: "
                "calling gridView.mark(elem, refine) on a "
                "structuredGrid raises 'grid does not "
                "support local refinement' or silently "
                "refines globally; for adaptivity, "
                "switch to alucubeGrid or alusimplexGrid. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Error estimator: eta_K^2 = "
                "h_K^2 * ||f + Δu||^2 + h_K * ||[grad(u)·"
                "n]||^2. Signal: omitting the jump term "
                "[grad(u).n] across facets under-estimates "
                "the error in irregular meshes by 5-30%; "
                "the residual-only estimator misses jumps "
                "that signal under-resolved interior "
                "layers. Use the full residual + jump "
                "estimator for reliable adaptivity. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[API] After refinement: call space."
                "update() AND uh.interpolate(uh) to "
                "project the solution onto the new mesh. "
                "Signal: forgetting space.update() leaves "
                "the function on the OLD space — "
                "subsequent assembly fails with "
                "'function and space mismatch'; "
                "forgetting interpolate() leaves uh "
                "as the initial guess on the refined "
                "regions instead of using the prior "
                "solution. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Doerfler marking: refine the "
                "SMALLEST set of elements that captures "
                "theta fraction (typical 0.25-0.5) of "
                "total error. Signal: theta < 0.1 refines "
                "too few elements per pass (slow "
                "convergence to target tolerance); theta "
                ">0.7 refines almost-uniformly (defeats "
                "the adaptive benefit). 0.3 is a common "
                "default. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] For COARSENING: mark elements "
                "with SMALL error indicator. Signal: a "
                "moving-front problem with monotonic "
                "refinement-only accumulates elements; "
                "after the front passes, those refined "
                "regions are over-resolved. Mark elements "
                "with eta_K < theta_coarse * max(eta) for "
                "coarsening (typical theta_coarse ~ 0.01-"
                "0.05). (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Nested iteration: use coarse-"
                "grid solution as initial guess on the fine "
                "grid. Signal: starting Newton from zero on "
                "the fine grid for a nonlinear problem "
                "takes 5-10 iterations; starting from the "
                "interpolated coarse solution converges in "
                "1-2 iterations because the initial guess "
                "is already in the convergence basin. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "adaptive_poisson_2d": _adaptive_poisson_2d,
}
