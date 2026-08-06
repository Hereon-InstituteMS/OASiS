"""NGSolve <-> scikit-fem: ALL FOUR role/position combinations.

THE CLAIM UNDER TEST. The sides table says of NGSolve and of scikit-fem alike:
"**Either side, in either subdomain.** All four role/position combinations were
run as real couplings on this install ... and all converged." Four is a stronger
claim than two, and it is the one row pair that makes it, so this fixture runs
four rather than the two a "both roles" claim would need.

Role and position are independent. A backend can be fine on the Dirichlet side
of the LEFT subdomain and wrong on the Dirichlet side of the RIGHT one: the
outward normal flips, and with it the sign of the exported flux and the sign of
the applied Neumann term. That is the whole reason a four-combination claim is
worth more than a two-combination one, and it is exactly what a `S = 1.0 if ...
else -1.0` line in a participant script gets wrong.

Each combination is checked against the closed form of the split conduction
problem — interface temperature, interface flux with the sign each side's own
outward normal implies, and conservation across the interface.
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
    L.require_available("ngsolve", "skfem")
    # position of each backend x which subdomain is Dirichlet = 4 combinations
    L.heat_arrangement("ngs_L_D_skf_R_N", "ngsolve", "skfem", "left")
    L.heat_arrangement("ngs_L_N_skf_R_D", "ngsolve", "skfem", "right")
    L.heat_arrangement("skf_L_D_ngs_R_N", "skfem", "ngsolve", "left")
    L.heat_arrangement("skf_L_N_ngs_R_D", "skfem", "ngsolve", "right")
    print("combinations_run=4")


L.main(body)
