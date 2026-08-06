"""Tier-2: the seven wall models nobody documents, on the circle geometry.

  surface_interaction:10  'adiabatic' takes NO arguments and transfers no energy
                          — the tallied etot is at round-off, like specular, not
                          merely small.
  surface_interaction:11  'transparent' is two independent flags. Transparent
                          ELEMENTS on a non-transparent MODEL are caught with a
                          clear message; the transparent MODEL on ordinary
                          elements is NOT, and the run dies in the collision
                          routine talking about cell volume.
  surface_interaction:13  cll needs five numbers and range-checks them; piston
                          needs an axis-aligned normal and says so only at the
                          start of the first run.

The energy claim is stated as a RATIO between two runs of the same deck — the
adiabatic tally against the diffuse tally on the identical geometry, seed and
run length — and the threshold is "many orders of magnitude", not a number: a
round-off residue is 15+ decades below a physical flux, so the test asks for a
ratio below 1e-6 and would still pass if the round-off residue were a thousand
times bigger than it is. Nothing is pinned.

Mutation control: T2_MUTATE=1 gives the adiabatic deck a DIFFUSE wall at the
same temperature the other decks use, changing nothing else, so the wall that is
supposed to be thermally inert starts exchanging energy:
`adiabatic_transfers_no_energy` and `adiabatic_and_specular_agree` go False. It
also flags the read_surf line of the transparent-model deck as 'transparent', so
the elements and the model finally agree and the cell-volume abort disappears,
sending `a_transparent_model_on_solid_elements_dies_in_the_collider` False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, find_example, run, skip_if_unavailable, stats_rows,
)

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
read_surf {circle}{rsopt}
surf_collide W {style}
surf_modify all collide W
collide vss air air.vss
fix in emit/face air xlo
timestep 0.0001
{extra}
stats 200
stats_style step np nscoll {cols}
run 400
"""

TALLY = ("compute cs surf all all etot\n"
         "fix fs ave/surf all 10 20 200 c_cs[1]\n"
         "compute rs reduce sum f_fs")


def go(style: str, rsopt: str = "", extra: str = "", cols: str = ""):
    rc, txt = run(DECK.format(circle=CIRCLE, rsopt=rsopt, style=style,
                              extra=extra, cols=cols), DATA)
    h, r = stats_rows(txt)
    return rc, errors(txt), (col(h, r, cols.split()[0]) if (cols and h and r) else [])


if MUTATE:
    print("mutation=adiabatic_wall_made_diffuse_and_transparent_elements_flagged")

ADIABATIC = "diffuse 300.0 1.0" if MUTATE else "adiabatic"

rc_d, e_d, q_d = go("diffuse 300.0 1.0", extra=TALLY, cols="c_rs")
rc_s, e_s, q_s = go("specular", extra=TALLY, cols="c_rs")
rc_a, e_a, q_a = go(ADIABATIC, extra=TALLY, cols="c_rs")

all_ran = all(rc == 0 for rc in (rc_d, rc_s, rc_a)) and not (e_d + e_s + e_a)
print(f"the_three_wall_models_all_run={all_ran}")

# The diffuse wall is the scale. Both inert walls must sit many decades below
# it. 1e-6 is a threshold with six decades of slack against a round-off residue.
scale = max(abs(v) for v in q_d[1:]) if len(q_d) > 1 else 0.0
spec = max(abs(v) for v in q_s[1:]) if len(q_s) > 1 else 0.0
adia = max(abs(v) for v in q_a[1:]) if len(q_a) > 1 else 0.0
print(f"the_diffuse_wall_tallies_energy={scale > 0.0}")
print(f"specular_transfers_no_energy={scale > 0.0 and spec / scale < 1e-6}")
print(f"adiabatic_transfers_no_energy={scale > 0.0 and adia / scale < 1e-6}")
print(f"adiabatic_and_specular_agree="
      f"{scale > 0.0 and adia / scale < 1e-6 and spec / scale < 1e-6}")
if scale > 0.0 and adia / scale >= 1e-6 and not MUTATE:
    print(f"UNEXPECTED: adiabatic/diffuse energy ratio is {adia / scale}")

rc_aa, e_aa, _ = go("adiabatic 300.0")
ADIA_MSG = ("ERROR: Illegal surf_collide adiabatic command "
            "(../surf_collide_adiabatic.cpp:37)")
print(f"adiabatic_takes_no_arguments={any(ADIA_MSG in e for e in e_aa)}")

# surface_interaction:11 — the two directions of the transparent pairing.
rc_t, e_t, _ = go("transparent", rsopt=" transparent" if MUTATE else "")
VOLUME_MSG = ("ERROR on proc 0: Collision cell volume is zero "
              "(../collide.cpp:441)")
model_only_dies = any(VOLUME_MSG in e for e in e_t)
print(f"a_transparent_model_on_solid_elements_dies_in_the_collider="
      f"{model_only_dies}")
if not model_only_dies and not MUTATE:
    print(f"UNEXPECTED: transparent-model deck printed {e_t}")

rc_te, e_te, _ = go("diffuse 300.0 1.0", rsopt=" transparent")
ELEM_MSG = ("transparent surface elements with invalid collision model or "
            "reaction model (../surf.cpp:397)")
print(f"the_other_direction_is_checked_and_named="
      f"{any(ELEM_MSG in e for e in e_te)}")

rc_tt, e_tt, _ = go("transparent", rsopt=" transparent")
print(f"both_flags_together_run={rc_tt == 0 and not e_tt}")

# surface_interaction:13 — cll and piston.
rc_c, e_c, _ = go("cll 300.0 1.0 1.0 1.0 1.0")
print(f"cll_with_five_numbers_runs={rc_c == 0 and not e_c}")
rc_c3, e_c3, _ = go("cll 300.0 1.0 1.0")
CLL_ARG = "ERROR: Illegal surf_collide cll command (../surf_collide_cll.cpp:54)"
print(f"cll_arg_count_is_checked={any(CLL_ARG in e for e in e_c3)}")
rc_cr, e_cr, _ = go("cll 300.0 1.5 1.0 1.0 1.0")
CLL_RANGE = ("ERROR: Surf_collide cll accommodation coeffs must be >= 0 and "
             "<= 1 (../surf_collide_cll.cpp:65)")
print(f"cll_coefficients_are_range_checked={any(CLL_RANGE in e for e in e_cr)}")

rc_p, e_p, _ = go("piston 100.0")
PISTON_MSG = ("ERROR: Surf_collide piston assigned to surface with non "
              "axis-aligned normal (../surf_collide_piston.cpp:76)")
piston_late = any(PISTON_MSG in e for e in e_p)
print(f"piston_rejects_a_curved_body={piston_late}")
# and it does so only after setup has printed, i.e. not at parse time
print(f"the_piston_check_is_not_at_parse_time="
      f"{piston_late and 'Created 100 child grid cells' not in ''.join(e_p)}")

rc_td, e_td, _ = go("td 300.0")
print(f"td_takes_a_wall_temperature={rc_td == 0 and not e_td}")

ok = (all_ran and scale > 0.0 and spec / scale < 1e-6 and adia / scale < 1e-6
      and any(ADIA_MSG in e for e in e_aa) and model_only_dies
      and any(ELEM_MSG in e for e in e_te) and rc_tt == 0 and not e_tt
      and rc_c == 0 and not e_c and any(CLL_ARG in e for e in e_c3)
      and any(CLL_RANGE in e for e in e_cr) and piston_late
      and rc_td == 0 and not e_td)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
