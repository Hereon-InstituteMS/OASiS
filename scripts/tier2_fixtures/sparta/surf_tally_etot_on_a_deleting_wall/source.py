"""Tier-2: `compute surf ... etot` on a `vanish` wall kills the process, with
no ERROR line anywhere.

  surface_interaction:12  the crash is specific to the etot keyword and to a
                          collision model that deletes the particle. Every other
                          compute surf keyword runs on the same wall, and etot
                          runs on every other wall.

This is the one failure mode in the SPARTA catalog that a driver checking for
the string 'ERROR' reports as clean, so the assertion is deliberately about the
EXIT STATUS and the ABSENCE of a message, not about any number.

The mechanism is in the source, and it is why the keyword matters:
SurfCollideVanish::collide and SurfCollideTransparent::collide are the only two
styles that leave the 'reaction' out-parameter unassigned — the argument is
unnamed in their signatures — where every other style writes reaction = 0 first;
ComputeSurf::surf_tally's ETOT branch then indexes surf->sr[isr] on that stale
value with no reaction model loaded. The fixture proves the consequence, not the
source line: it runs the whole keyword list on both walls and shows exactly one
cell of that grid dies.

Mutation control: T2_MUTATE=1 replaces the vanish wall with a diffuse wall at
the same temperature the control deck uses — the single edit that removes the
deleting collision model — leaving the compute, the fix, the geometry, the seed
and the run length identical. Nothing then crashes and
`etot_on_a_vanish_wall_dies_on_a_signal` and
`it_dies_with_no_error_line_at_all` both go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, find_example, run, skip_if_unavailable  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("air.species", "air.vss")
CIRCLE = find_example("circle", "data.circle")
if CIRCLE is None:
    print("SKIP: SPARTA examples/circle/data.circle not found (set SPARTA_ROOT)")
    sys.exit(0)

DECK = """seed 12345
dimension 2
global gridcut 0.0
boundary o r p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
global nrho 1.0 fnum 0.001
species air.species N O
mixture air N O vstream 100.0 0 0
read_surf {circle}
surf_collide W {style}
surf_modify all collide W
collide vss air air.vss
fix in emit/face air xlo
timestep 0.0001
compute cs surf all all {kw}
fix fs ave/surf all 10 20 200 c_cs[1]
stats 200
run 400
"""

KEYWORDS = ["n", "nwt", "nflux", "mflux", "press", "shx", "ke", "erot",
            "evib", "etot", "fx", "fy"]

# The pathology is the DELETING collision model. Under mutation it becomes an
# ordinary diffuse wall; nothing else in the deck moves.
DELETING = "diffuse 300.0 1.0" if MUTATE else "vanish"
CONTROL = "diffuse 300.0 1.0"

if MUTATE:
    print("mutation=vanish_wall_replaced_by_a_diffuse_wall")


def go(style: str, kw: str):
    return run(DECK.format(circle=CIRCLE, style=style, kw=kw), DATA)


del_rc, del_err = {}, {}
ctl_rc = {}
for kw in KEYWORDS:
    rc, txt = go(DELETING, kw)
    del_rc[kw], del_err[kw] = rc, errors(txt)
    rc2, _ = go(CONTROL, kw)
    ctl_rc[kw] = rc2

# A negative return code is a signal, not an exit status SPARTA chose.
signalled = [k for k, rc in del_rc.items() if rc < 0]
print(f"etot_on_a_vanish_wall_dies_on_a_signal={signalled == ['etot']}")
if signalled != ["etot"]:
    print(f"UNEXPECTED: signalled keywords on the deleting wall: {signalled}")

print(f"it_dies_with_no_error_line_at_all="
      f"{'etot' in del_rc and del_rc['etot'] < 0 and not del_err['etot']}")

print(f"every_other_keyword_survives_the_same_wall="
      f"{all(del_rc[k] == 0 for k in KEYWORDS if k != 'etot')}")

print(f"etot_is_fine_on_an_ordinary_wall={ctl_rc.get('etot') == 0}")
print(f"the_whole_keyword_list_is_fine_on_an_ordinary_wall="
      f"{all(rc == 0 for rc in ctl_rc.values())}")

# The documented workaround: ke + erot + evib in one compute, on the same wall.
rc_w, txt_w = run(
    DECK.format(circle=CIRCLE, style=DELETING, kw="ke erot evib"), DATA)
print(f"ke_erot_evib_together_are_the_workaround={rc_w == 0 and not errors(txt_w)}")

# A vanish wall with no surface tally at all runs perfectly well, so the wall
# model is not the problem on its own — the pairing is.
NO_TALLY = DECK.replace("compute cs surf all all {kw}\n", "").replace(
    "fix fs ave/surf all 10 20 200 c_cs[1]\n", "")
rc_n, txt_n = run(NO_TALLY.format(circle=CIRCLE, style=DELETING), DATA)
print(f"the_deleting_wall_alone_is_harmless={rc_n == 0 and not errors(txt_n)}")

ok = (signalled == ["etot"] and not del_err["etot"]
      and all(del_rc[k] == 0 for k in KEYWORDS if k != "etot")
      and all(rc == 0 for rc in ctl_rc.values())
      and rc_w == 0 and not errors(txt_w)
      and rc_n == 0 and not errors(txt_n))
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
