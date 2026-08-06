"""Tier-2: `fix field/grid` on its own is a no-op, and the activating line
takes a FIX ID that upstream's example makes look like a flag.

  rarefied_flow:15  the body force needs three lines that agree — a grid-style
                    (or particle-style) variable, the fix, and a 'global field'
                    line naming the FIX ID. With the global line missing the
                    deck runs to completion with the particle statistics
                    bit-identical to a deck carrying no field at all.

The no-op half is proved by identity against a control deck, which is the only
honest way to assert that something did nothing. The active half is proved by
the same box with the global line present, where the temperature leaves its
initial value; that is a direction, not a magnitude, so no number is pinned.

Upstream's examples/bfield deck names its fix '1', so its activating line reads
'global field grid 1 0' and copies as though the 1 were a boolean. A fix with
any other ID and that same line aborts.

Mutation control: T2_MUTATE=1 adds the missing 'global field grid <fixID> 0'
line to the deck whose whole point is that it lacks one, changing nothing else —
same variable, same fix, same box, grid, seed and run length. The field then
acts, the run stops matching the control, and `a_field_fix_without_the_global_
line_is_a_no_op` goes False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

BOX = """seed 12345
dimension 2
boundary p p p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e11
create_particles gas n 0
timestep 1e-8
compute gt temp
{body}
stats 100
stats_style step np c_gt
run 300
"""


def go(body: str):
    rc, txt = run(BOX.format(body=body), DATA)
    h, r = stats_rows(txt)
    temps = col(h, r, "c_gt") if (h and r) else []
    return rc, errors(txt), temps


VAR_AND_FIX = "variable ax grid 1e10\nfix ff field/grid ax NULL NULL\n"
ACTIVATE = "global field grid ff 0\n"

if MUTATE:
    print("mutation=global_field_line_added_to_the_deck_that_was_missing_it")

rc_ctl, e_ctl, t_ctl = go("")
rc_off, e_off, t_off = go(VAR_AND_FIX + (ACTIVATE if MUTATE else ""))
rc_on, e_on, t_on = go(VAR_AND_FIX + ACTIVATE)

print(f"the_control_deck_runs={rc_ctl == 0 and not e_ctl and len(t_ctl) > 1}")

# The no-op: identical temperature column, row for row, against the control.
noop = (rc_off == 0 and not e_off and bool(t_ctl) and t_off == t_ctl)
print(f"a_field_fix_without_the_global_line_is_a_no_op={noop}")
if not noop and not MUTATE:
    print(f"UNEXPECTED: no-global deck gave {t_off[-1:]} against control {t_ctl[-1:]}")

# The control deck is at equilibrium, so its temperature column is flat; the
# activated one must leave it. A direction, not a magnitude.
control_is_flat = bool(t_ctl) and all(v == t_ctl[0] for v in t_ctl)
field_acts = (rc_on == 0 and not e_on and bool(t_on)
              and t_on[0] == t_ctl[0] and t_on[-1] > t_ctl[-1])
print(f"the_control_deck_is_flat={control_is_flat}")
print(f"the_activated_field_heats_the_gas={field_acts}")

# 'global field grid <fixID>' really is a fix ID: the upstream-looking numeric
# form fails against a fix named anything but that number.
rc_id, e_id, _ = go(VAR_AND_FIX + "global field grid 1 0\n")
ID_MSG = "ERROR: External field fix ID not found (../update.cpp:221)"
print(f"the_global_field_argument_is_a_fix_id={any(ID_MSG in e for e in e_id)}")

# The variable style is checked, and the name must be bare.
rc_eq, e_eq, _ = go("variable ax equal 1e10\n"
                    "fix ff field/grid ax NULL NULL\n" + ACTIVATE)
STYLE_MSG = ("ERROR: Variable for fix field/grid is invalid style "
             "(../fix_field_grid.cpp:105)")
print(f"an_equal_style_variable_is_rejected={any(STYLE_MSG in e for e in e_eq)}")
rc_pf, e_pf, _ = go("variable ax grid 1e10\n"
                    "fix ff field/grid v_ax NULL NULL\n" + ACTIVATE)
NAME_MSG = ("ERROR: Variable name for fix field/grid does not exist "
            "(../fix_field_grid.cpp:103)")
print(f"a_v_prefixed_name_is_not_found={any(NAME_MSG in e for e in e_pf)}")

ok = (rc_ctl == 0 and not e_ctl and noop and control_is_flat and field_acts
      and any(ID_MSG in e for e in e_id)
      and any(STYLE_MSG in e for e in e_eq)
      and any(NAME_MSG in e for e in e_pf))
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
