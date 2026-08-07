"""FEniCSx <-> deal.II exchanging a VECTOR across the interface.

The deal.II side is the interesting one. deal.II has no Python API, so its
participant is TWO files — a compiled C++ solver plus a thin Python wrapper —
and unlike every other backend there is a BUILD STEP before a coupling can run
at all. The scalar (heat) solver has been exercised that way; the vector one
(elast_iface_dealii.cc, plane-strain elasticity with an FESystem of two FE_Q
components) is built and run here, so the claim that deal.II participates in
vector coupling rests on a run rather than on the existence of a source file.

Two things the C++ side has to get right that the Python participants get for
free, and that this fixture is what checks:

  * the two INTERFACE CORNERS must stay governed by the outer boundary on both
    sides. In deal.II that is a std::map merge whose semantics (insert does not
    overwrite) decide it; get it backwards and the Neumann subproblem is still
    well posed, still converges, and lands a few percent off;
  * the exported traction must use the shipped convention -(sigma . n_own), so
    that the two sides cancel and the Neumann side applies the partner's
    numbers unchanged.

Everything asserted is componentwise; see vector_pair_fenics_skfem for what the
checks are and why convergence is not one of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402


def body() -> None:
    L.require_available("fenics", "dealii")
    # Building the shipped vector solver is part of the claim, so do it first
    # and say where it landed: a fixture that silently reused another
    # checkout's binary would be verifying the wrong tree, which is exactly
    # what the shared build directory used to allow.
    exe = L.dealii_exe("elast_iface_dealii")
    print(f"dealii_vector_solver_built={exe}")
    L.vector_arrangement("dealii_D_fenics_N", "dealii", "fenics", "left")
    L.vector_arrangement("fenics_D_dealii_N", "fenics", "dealii", "left")
    print("vector_arrangements_run=2")


L.main(body)
