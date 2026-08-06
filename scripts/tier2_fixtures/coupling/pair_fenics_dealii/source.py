"""FEniCSx <-> deal.II, both roles.

THE CLAIM UNDER TEST: the sides table gives deal.II an unstarred "yes" in both
columns, "coupled to FEniCSx and DUNE-fem, both roles".

deal.II is the only participant that does not exist until you build it. Its
payload says "THE PARTICIPANT IS TWO FILES: a compiled C++ solver and a thin
Python wrapper", ships `heat_iface_dealii.cc` and a `CMakeLists.txt` next to the
script, and gives the cmake invocation. So this fixture builds the shipped
source before it couples anything: if the shipped C++ does not compile against
this install's deal.II, the claim that a deal.II participant is available here
is false, and the fixture says so rather than skipping.

The payload's silent-wrong warning is specific and worth naming: "THE NODAL FLUX
MUST BE AVERAGED OVER BOTH ADJACENT CELLS. Assembling the interface flux cell by
cell and writing it into a per-node array is last-writer-wins, which silently
biases every interior interface node toward one cell." A biased nodal flux still
converges. It shows up as an interface flux that is off by a factor related to
the cell size — which the closed-form comparison here catches and a convergence
check does not.
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
    L.require_available("fenics", "dealii")
    exe = L.dealii_exe()          # builds the shipped .cc once, or raises
    print(f"dealii_solver_built={exe.is_file()}")
    L.heat_arrangement("dealii_D_fenics_N", "dealii", "fenics", "left")
    L.heat_arrangement("fenics_D_dealii_N", "fenics", "dealii", "left")
    print("pairs_run=2")


L.main(body)
