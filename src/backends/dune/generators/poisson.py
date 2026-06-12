"""DUNE-fem Poisson equation generators and knowledge."""


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]², u=0 on boundary — DUNE-fem with UFL."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    order = params.get("order", 1)
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², u=0 on boundary — DUNE-fem (UFL)"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})
u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dx
b = {f_val} * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="solution")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
max_val = float(vals.max())
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{len(vals)}}")

gridView.writeVTK("result", pointdata={{"phi": uh}})

summary = {{"max_value": max_val, "n_dofs": len(vals), "element_type": "Q1 quad"}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DUNE-fem Poisson solve complete.")
'''


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson with DUNE-fem using UFL (same forms as FEniCS)",
        "solver": "galerkin([a == b, dbc], solver='cg') — Newton-Krylov internally",
        "spaces": "lagrange(gridView, order=k) — Lagrange any order",
        "mesh": "structuredGrid (YaspGrid), ALUGrid (adaptive), Gmsh import",
        "pitfalls": [
            (
                "[API] DUNE-fem uses UFL — same syntax as "
                "FEniCS/dolfinx. Weak forms are largely "
                "interchangeable. Signal: a UFL form that "
                "compiles cleanly in dolfinx will also "
                "compile in DUNE-fem, but the FUNCTION SPACE "
                "construction differs: dune.fem.space.lagrange("
                "gridView, order=k) vs dolfinx.fem."
                "functionspace(domain, ('Lagrange', k)). "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Performance] First DUNE-fem run triggers "
                "JIT compilation of C++ code — 30-60s "
                "overhead. Signal: a 'simple' Poisson "
                "solve takes a minute on first run but "
                "milliseconds on the second; ~/.dune/dune-"
                "py/python/dune/generated/ holds the cache. "
                "Time profilers expose the JIT phase as "
                "module compilation. (Audit 2026-06-02.)"
            ),
            (
                "[Performance] Subsequent runs use CACHED "
                "compiled code — much faster. Signal: "
                "deleting ~/.dune/dune-py/ forces full "
                "recompile and resets the 30-60s overhead. "
                "If you change the UFL form text (even "
                "trivially), it triggers a recompile on "
                "first invocation of the new form. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Use dune.ufl.DirichletBC (NOT "
                "dolfinx.fem.dirichletbc). Signal: "
                "importing DirichletBC from dolfinx in a "
                "DUNE-fem script raises ImportError or "
                "wrong-signature TypeError — the DUNE "
                "constructor takes (space, value), not the "
                "dolfinx (V, value, dofs) triple. Use "
                "from dune.ufl import DirichletBC. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] VTK output: gridView.writeVTK('name', "
                "pointdata={'field': uh}). Signal: writing "
                "with the dolfinx io.VTXWriter / XDMFFile "
                "API fails — dune.grid gridView has its own "
                "writeVTK method (NOT dolfinx). The galerkin "
                "lagrange space's GridFunction is written "
                "via gridView.writeVTK; outputs .vtu "
                "directly readable in ParaView. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] structuredGrid creates QUAD "
                "elements, NOT triangles. Signal: a "
                "trial form that assumes triangular "
                "topology (e.g. relies on barycentric "
                "coordinates) compiles but the assembled "
                "matrix has wrong sparsity — quad "
                "elements have 4-node stencil vs 3-node "
                "for tri. For triangles, use Gmsh import "
                "or dune.alugrid. (Audit 2026-06-02.)"
            ),
            (
                "[API] Constant(value) needs a domain in "
                "newer UFL — use scalar literal directly: "
                "1.0 * v * dx. Signal: writing "
                "Constant(1.0) * v * dx without "
                "Constant(domain, 1.0) raises 'Constant "
                "requires domain' from UFL; the simplest "
                "fix is scalar arithmetic — Python "
                "scalars are accepted by UFL operator "
                "overloading. (Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
}
