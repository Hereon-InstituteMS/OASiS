"""DUNE-fem Poisson equation generators and knowledge.

The pitfalls in this module come from two different evidence levels and
say so in their provenance note:

  * "(Audit 2026-06-02.)"      — established by reading sources/docs.
  * "(Executed 2026-08-03 ...)" — a script was run against the
    installed dune-fem 2.12.0.2 and the quoted numbers and messages
    are that run's output. Those entries live in
    ``verified_api.EXECUTED_PITFALLS`` and are appended to the list
    below; see that module for the full measured API surface.
"""

from .verified_api import EXECUTED_PITFALLS


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
                "JIT compilation of C++ code. Signal: a "
                "'simple' Poisson solve prints 'DUNE-INFO: "
                "Compiling <X> (new)' on stderr and then "
                "produces no output for tens of seconds to "
                "minutes; the second run is milliseconds. "
                "The cache is <sys.prefix>/.cache/dune-py "
                "(inside a venv/conda env) or "
                "~/.cache/dune-py, NOT ~/.dune — see "
                "dune.packagemetadata.getDunePyDir(). "
                "(Corrected by execution 2026-08-03 on "
                "dune-fem 2.12.0.2: getDunePyDir() returned "
                "the conda env's .cache/dune-py and ~/.dune "
                "does not exist; the earlier ~/.dune claim "
                "was wrong for this version.)"
            ),
            (
                "[Performance] Subsequent runs use CACHED "
                "compiled code — much faster. Signal: "
                "deleting the dune-py cache directory forces "
                "a full recompile and resets the cost. "
                "MEASURED to cost a new build: a different "
                "Lagrange order, a different grid class, a "
                "direct instead of a Krylov solver, and ANY "
                "change to the form — including changing a "
                "bare float literal inside it (7.3125 -> "
                "9.8125 cost 25.204 s and added one .so). "
                "MEASURED free: re-running the identical form "
                "(0.051 s, no new .so) and changing the "
                ".value of a dune.ufl.Constant (0.00056 s, no "
                "new .so). (Audit 2026-06-02; rebuild "
                "triggers measured by execution 2026-08-03.)"
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
                "[API] structuredGrid creates CUBE "
                "elements (quadrilateral in 2D, hexahedron "
                "in 3D), NOT simplices, and dune.grid has "
                "no option that yields simplices. Signal: "
                "count the cells — an n-by-n domain gives "
                "n^2 elements through structuredGrid and "
                "2*n^2 through aluConformGrid / "
                "aluSimplexGrid. Do NOT ask the space: its "
                "UFL cell always says 'triangle'. For "
                "simplices use dune.alugrid (or a Gmsh "
                "import through it). (Executed 2026-08-03 "
                "on dune-fem 2.12.0.2: "
                "structuredGrid([0,0],[1,1],[4,4]) -> 16 "
                "elements of type 'quadrilateral'; "
                "aluConformGrid on the same "
                "cartesianDomain -> 32 of type 'triangle'. "
                "The earlier 'wrong sparsity' wording was "
                "dropped — nothing about the assembled "
                "matrix is wrong, the mesh is simply made "
                "of different cells.)"
            ),
            (
                "[API] Use dune.ufl.Constant(value, "
                "name=...), NOT ufl.Constant. Signal: plain "
                "ufl.Constant needs a domain and fails "
                "without one; dune.ufl.Constant takes a "
                "bare value and — unlike a Python float "
                "baked into the form — is RUNTIME-"
                "UPDATABLE: assigning c.value = ... and "
                "calling scheme.solve() again reuses the "
                "compiled module instead of triggering a "
                "JIT rebuild. A Python scalar literal also "
                "works but re-compiles whenever it changes. "
                "(Audit 2026-06-02; the dune.ufl.Constant "
                "correction and the reuse behaviour "
                "verified by execution 2026-08-03.)"
            ),
            *EXECUTED_PITFALLS,
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
}
