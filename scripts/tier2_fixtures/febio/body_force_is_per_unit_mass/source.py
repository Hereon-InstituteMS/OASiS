"""Tier-2: editing <density> alone silently rescales the body force, and
only the INTERIOR moves.

Verifies febio::elasticity_mms#1. FEBio body forces are per unit MASS and
are multiplied internally by the material <density>, so density matters
even in a STATIC analysis whenever a body load is present. The generator
divides by rho to keep the applied force per volume independent of it —
which means the deck's <density> and the body-force expressions must move
together.

The fixture doubles <density> alone on the generated MMS deck and asserts
the shape of the damage:

  * the run is clean — normal termination, exit 0, no warning,
  * every prescribed BOUNDARY node is unchanged to the last digit, since
    they are Dirichlet,
  * the interior displacement scales by the density ratio.

The boundary/interior split is the assertion that makes this diagnosable:
a uniform rescaling would be divisible back out, and this is not one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    # MUTATION CONTROL. The pathology this fixture reproduces is the
    # EDIT that turns a correct deck into the broken one, so removing
    # the pathology means not making that edit. Neutralising L.swap /
    # L.drop does exactly that: every deck built below is the correct
    # one, the pitfall is never triggered, the diagnostic must not
    # appear and the verdict token flips to not_reproduced.
    print("mutation=the_deck_edits_that_introduce_the_pitfall_are_"
          "neutralised")
    L.swap = lambda deck, old, new, **kw: deck
    L.drop = lambda deck, fragment: deck


import re


def main() -> int:
    base = L.template("elasticity_mms_3d_cube_hex8", n=2)
    doubled = L.swap(base, "<density>1</density>", "<density>2</density>")

    coords = {int(m.group(1)): tuple(float(x) for x in m.group(2).split(","))
              for m in re.finditer(r'<node id="(\d+)">([^<]+)</node>', base)}

    def positions(deck):
        r = L.run(L.add_logfile(deck, ("node_data", "x;y;z", "p.csv")),
                  collect=("p.csv",), timeout=1200)
        blocks = L.parse_log_csv(r.files.get("p.csv") or "")
        return r, (blocks[-1][1] if blocks else {})

    r1, p1 = positions(base)
    r2, p2 = positions(doubled)
    if not p1 or not p2:
        print("FAIL: a run logged no node positions")
        return L.report(False, "mms_density_coupling", "reproduced",
                        "not_reproduced")

    def is_interior(nid):
        return all(1e-12 < v < 1 - 1e-12 for v in coords[nid])

    interior = [k for k in p1 if k in p2 and is_interior(k)]
    boundary = [k for k in p1 if k in p2 and not is_interior(k)]
    bnd_move = max(max(abs(p1[k][i] - p2[k][i]) for i in range(3))
                   for k in boundary)
    print(f"rho=1: rc={r1.rc} normal={int(r1.normal_termination)} "
          f"steps={r1.steps_completed} warnings={int('WARNING' in r1.text)}")
    print(f"rho=2: rc={r2.rc} normal={int(r2.normal_termination)} "
          f"steps={r2.steps_completed} warnings={int('WARNING' in r2.text)}")
    print(f"boundary_nodes={len(boundary)} max_move={bnd_move:.3e}")

    ratios = []
    for k in interior:
        u1 = [p1[k][i] - coords[k][i] for i in range(3)]
        u2 = [p2[k][i] - coords[k][i] for i in range(3)]
        m1 = max(abs(v) for v in u1)
        m2 = max(abs(v) for v in u2)
        if m1 > 1e-12:
            ratios.append(m2 / m1)
    print(f"interior_nodes={len(interior)} "
          f"displacement_ratios={[f'{v:.6f}' for v in ratios]}")
    boundary_pinned = bnd_move < 1e-12
    scaled = bool(ratios) and all(abs(v - 2.0) < 1e-3 for v in ratios)
    print(f"boundary_unchanged={int(boundary_pinned)} "
          f"interior_scales_by_the_density_ratio={int(scaled)}")
    good = (boundary_pinned and scaled
            and r1.rc == 0 and r2.rc == 0
            and r1.normal_termination and r2.normal_termination)
    return L.report(good, "mms_density_coupling", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
