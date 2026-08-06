"""FEniCSx <-> scikit-fem: does the pair the sides table claims actually solve
the problem, in both role/position arrangements?

THE CLAIM UNDER TEST (src/tools/coupling_knowledge.py, the sides table plus the
FEniCSx and scikit-fem payloads): FEniCSx can take either side, scikit-fem can
take either side "in either subdomain", the two were coupled to each other on
this install with non-matching interface meshes, and the run CONVERGED.

Why convergence is not the assertion. A partitioned fixed-point iteration
converges to a FIXED POINT. That is the solution only if the two participants
exchange the right quantity, with the right sign, in the right units. Get any of
those wrong and the loop still converges — smoothly, and with a perfect flux
balance. Flipping the sign of the exported flux in the shipped scikit-fem script
makes this run converge in 62 iterations, balance to 4e-08, and land 22 K away
from the exact interface temperature. So the assertions are against the closed
form of the split conduction problem, not against the driver's verdict.

Nothing here is pinned. The tolerances are what the physics has to beat; the
errors are printed so the run reports its own numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def body() -> None:
    L.require_available("fenics", "skfem")
    p = L.DEFAULT
    # theta = 1/(1+rho), which the knowledge says to compute before running
    # anything; `heat_arrangement` takes it from the problem when not given.
    L.heat_arrangement("fenics_D_skfem_N", "fenics", "skfem", "left")
    L.heat_arrangement("skfem_D_fenics_N", "skfem", "fenics", "left")
    # …and with the roles swapped, so FEniCSx is proven on BOTH sides against
    # this partner rather than only in one arrangement.
    L.heat_arrangement("fenics_N_skfem_D", "fenics", "skfem", "right")
    print(f"rho_dirichlet_left={p.rho('left'):.6f}")
    print(f"rho_dirichlet_right={p.rho('right'):.6f}")
    print("pairs_run=3")


L.main(body)
