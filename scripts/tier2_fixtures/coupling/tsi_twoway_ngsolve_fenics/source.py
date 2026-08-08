"""TWO-WAY TSI ACROSS A SECOND PAIR OF CODES: NGSolve with FEniCSx, and NGSolve
with scikit-fem.

THE CLAIM UNDER TEST: a second, independent cross-code pair runs the same
two-way thermo-structural coupling, so the capability is a property of the
coupling path rather than of one pairing that happened to work.

NGSolve IS THE HARD ONE, DELIBERATELY. Its mesh comes out of netgen and is
UNSTRUCTURED, while dolfinx and scikit-fem both use a tensor-product
triangulation here. So in these arrangements the exchange is between a regular
grid of points and an irregular cloud, in both directions, and no node can
coincide with a partner node by accident. Whatever the mesh-to-mesh map gets
wrong has nowhere to hide behind a lucky alignment.

It is also a different discretisation of the structural half: VectorH1 of order
2 with per-component Dirichlet boundaries (`dirichletx=` / `dirichlety=`) rather
than a collapsed sub-space.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                          # noqa: E402
import tsilib as T                                               # noqa: E402


def body() -> None:
    L.require_available("ngsolve", "fenics", "skfem")

    T.full_pair_check("ngsolve_T_fenics_M", "ngsolve", "fenics")
    T.full_pair_check("skfem_T_ngsolve_M", "skfem", "ngsolve")

    print("pairs_run=2")


L.main(body)
