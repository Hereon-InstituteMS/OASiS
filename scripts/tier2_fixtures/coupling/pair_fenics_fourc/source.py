"""FEniCSx <-> 4C, both directions. The flagship pair.

THE CLAIM UNDER TEST: the sides table gives 4C an unstarred "yes" in BOTH
columns — "coupled to FEniCSx, both roles" — and the 4C payload adds that this
"contradicts the deprecated `coupled_solve` docstring, which lists 4C on the
Neumann side only". That contradiction is the point: one of the two statements
about 4C in this repo is wrong, and until now nothing decided which.

This is also the pair with the most ways to converge to a wrong answer, and the
4C payload lists them:
  * a spatially varying 4C boundary datum is `VAL x FUNCT`, and FUNCT is a
    symbolic expression, not a table — so the imported samples are fitted by a
    least-squares polynomial. A bad fit is a silently wrong boundary condition,
    not an error.
  * VTU output lands in `scatra-<step>-<rank>.vtu` and THE TRAILING NUMBER IS
    THE RANK. Sorting on it returns the INITIAL CONDITION, an all-zero field
    that reads as a converged solve of a trivial problem.
  * a 4C QUAD4 VTU repeats every node once per element, so a participant that
    does not collapse duplicates exports four times too many points.
Each of those produces a run that converges. None is visible without checking
the interface value and flux against the closed form, which is what this does.

The 4C participant is a WRAPPER: `command` runs a plain Python with numpy and
meshio, and the 4C binary path goes into a constant INSIDE the script. That is
the distinction four dead `.replace()` calls used to drop from the served launch
text, and this fixture exercises the corrected arrangement by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402


def body() -> None:
    L.require_available("fenics", "fourc")
    # 4C on the DIRICHLET side — the role the deprecated docstring says it
    # cannot take.
    L.heat_arrangement("fourc_D_fenics_N", "fourc", "fenics", "left")
    # …and on the Neumann side, the role that docstring does allow.
    L.heat_arrangement("fenics_D_fourc_N", "fenics", "fourc", "left")
    # …and with 4C in the RIGHT subdomain on the Dirichlet side, so its
    # outward normal points the other way.
    L.heat_arrangement("fenics_N_fourc_D", "fenics", "fourc", "right")
    print("pairs_run=3")


L.main(body)
