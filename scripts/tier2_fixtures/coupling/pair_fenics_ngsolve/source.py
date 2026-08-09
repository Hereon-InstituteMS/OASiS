"""FEniCSx <-> NGSolve, both roles.

THE CLAIM UNDER TEST: the sides table lists FEniCSx as "coupled to 4C, NGSolve,
scikit-fem, both roles" and NGSolve as coupled "to FEniCSx and scikit-fem, all
four role/position combinations". This fixture is the FEniCSx-NGSolve half of
both rows.

NGSolve earns extra scrutiny here because two of its documented traps are
silently-wrong rather than loud, and both would show up as a converged coupling
with the wrong answer: two consecutive `gfu.Set(..., definedon=...)` calls cancel
each other, so the outer Dirichlet value and the interface value cannot both be
set that way; and `Integrate(grad(gfu)[0], ..., definedon=mesh.Boundaries(...))`
returns exactly 0.0, because an H1 GridFunction's gradient has no boundary trace.
A participant with either defect converges. Only a comparison against the closed
form separates it from one without.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The shared library lives in a sibling `_lib/` DIRECTORY, not as a bare
# file, because scripts/mutate_tier2_fixtures.py stages a fixture into a
# scratch tree and copies only sibling directories whose name starts with
# `_`. As a bare file it would not be copied, the staged fixture could not
# import it, and every mutation verdict would be VACUOUS_BASELINE.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402


def body() -> None:
    L.require_available("fenics", "ngsolve")
    L.heat_arrangement("fenics_D_ngsolve_N", "fenics", "ngsolve", "left")
    L.heat_arrangement("ngsolve_D_fenics_N", "ngsolve", "fenics", "left")
    L.heat_arrangement("fenics_N_ngsolve_D", "fenics", "ngsolve", "right")
    print("pairs_run=3")


L.main(body)
