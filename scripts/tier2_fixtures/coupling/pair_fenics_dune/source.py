"""FEniCSx <-> DUNE-fem, both roles.

THE CLAIM UNDER TEST: the sides table gives DUNE-fem an unstarred "yes" in both
columns, "coupled to FEniCSx and deal.II, both roles ... and all converged".

DUNE is the backend whose payload makes a claim about the RUN, not only about
the answer: the participant is a fresh process every coupling iteration, and
DUNE-fem JIT-compiles each distinct UFL form on first use, so "if the FORM TEXT
changes with the imported data you pay that compile on every iteration and the
coupling appears to hang". The shipped script is supposed to keep the form
structurally constant and push the imported data into a discrete function's dof
vector instead. A coupling that finishes inside a bounded wall-clock budget is
the observable form of that claim, so this fixture runs with a real timeout
rather than an unbounded one.

The payload also states that DUNE's exported flux "carries small solver noise"
because the default `galerkin(..., solver="cg")` projection runs at a loose
linear tolerance — "far below a 1e-8 coupling tolerance but not machine
precision". That is a claim about the SIZE of an error, so the flux tolerance
here is the loosest of any pair fixture, and the measured error is printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def body() -> None:
    L.require_available("fenics", "dune")
    L.heat_arrangement("dune_D_fenics_N", "dune", "fenics", "left")
    L.heat_arrangement("fenics_D_dune_N", "fenics", "dune", "left")
    print("pairs_run=2")


L.main(body)
