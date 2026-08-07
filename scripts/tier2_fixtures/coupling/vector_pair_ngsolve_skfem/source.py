"""NGSolve <-> scikit-fem exchanging a VECTOR across the interface.

The third backend pair, and the one that carries an API hazard the other two do
not: NGSolve's VectorH1 is BLOCKED BY COMPONENT. GetDofNrs(NodeId(VERTEX, i))
returns the u_x and u_y dofs of vertex i and those two indices are nv apart,
not adjacent — the opposite of dolfinx, where a blocked space puts component c
of node n at n*bs + c. Writing nodal values with the wrong adjacency assumption
scatters u_y into the u_x block: the vector has the right length, the solve
runs, the coupling converges, and every number is wrong. It is exactly the
class of mistake a scalar fixture cannot produce.

NGSolve also meshes with netgen rather than on a tensor grid, so its interface
node count is whatever the mesher gives — the non-matching interface is not
arranged, it is unavoidable.

Everything asserted is componentwise; see vector_pair_fenics_skfem for what the
checks are and why convergence is not one of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402


def body() -> None:
    L.require_available("ngsolve", "skfem")
    L.vector_arrangement("ngsolve_D_skfem_N", "ngsolve", "skfem", "left")
    L.vector_arrangement("skfem_D_ngsolve_N", "skfem", "ngsolve", "left")
    L.vector_arrangement("ngsolve_N_skfem_D", "ngsolve", "skfem", "right",
                         problem=L.VECTOR_MIRRORED)
    print("vector_arrangements_run=3")


L.main(body)
