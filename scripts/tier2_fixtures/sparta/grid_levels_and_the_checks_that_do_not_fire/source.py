"""Tier-2: the multi-level create_grid syntax, and the two diagnostics that
report health on decks that have none.

  adaptive_grid:6  'create_grid ... levels <N>' plus a 'region' or 'subset'
                   clause for EVERY level 2..N. There is no 'level' keyword —
                   an earlier catalog entry gave one and it does not parse.
  adaptive_grid:7  'fix grid/check' is an internal-consistency assertion, not a
                   setup check: it stays at exactly zero on a deck whose
                   timestep makes the per-step collision count comparable to the
                   particle count. 'outside yes' is opt-in and all three modes
                   tally into f_ID.
  adaptive_grid:8  'fix balance' on a serial build reports imbalance exactly 1
                   forever, so a 1 there says nothing about the load.
  adaptive_grid:9  'fix move/surf' pushed too hard fails inside the CUTTING
                   code with a message that names neither the fix nor the step
                   size, and grid/check does not catch it either.

The grid/check claim is a NEGATIVE one, so it is stated the only way a negative
can be: the counter is exactly 0, on a deck whose own collision column shows it
is nonsense, and the warn-mode warning string never appears. The absurd deck is
identified as absurd by comparison with a sane one in the same script, not by a
number.

Mutation control: T2_MUTATE=1 repairs the create_grid keyword ('level' becomes
'levels') and fills in the level that was left unset, so both parse failures
disappear and `there_is_no_level_keyword` and `every_declared_level_must_be_set`
go False. Nothing else moves.
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

DATA = skip_if_unavailable("ar.species", "ar.vss")

GRID = """seed 12345
dimension 2
boundary p p p
create_box 0 1e-3 0 1e-3 -0.5 0.5
region inner block 0.25e-3 0.75e-3 0.25e-3 0.75e-3 INF INF
create_grid {cg}
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e11
create_particles gas n 0
timestep 1e-8
stats 10
stats_style step np ngrid maxlevel
run 10
"""


def grid(cg: str):
    rc, txt = run(GRID.format(cg=cg), DATA)
    h, r = stats_rows(txt)
    return rc, errors(txt), (col(h, r, "Maxlevel") if (h and r) else [])


if MUTATE:
    print("mutation=create_grid_keyword_repaired_and_the_unset_level_filled_in")

# adaptive_grid:6 — the working forms, and the two ways to get it wrong.
rc_r, e_r, ml_r = grid("10 10 1 levels 2 region 2 inner 2 2 1")
rc_s, e_s, ml_s = grid("10 10 1 levels 2 subset 2 3*7 3*7 * 2 2 1")
rc_3, e_3, ml_3 = grid("10 10 1 levels 3 region 2 inner 2 2 1 region 3 inner 2 2 1")
print(f"levels_plus_region_builds_a_hierarchy={rc_r == 0 and not e_r and ml_r[:1] == [2.0]}")
print(f"levels_plus_subset_builds_a_hierarchy={rc_s == 0 and not e_s and ml_s[:1] == [2.0]}")
print(f"a_third_level_is_reachable={rc_3 == 0 and not e_3 and ml_3[:1] == [3.0]}")

BAD_KEYWORD = ("10 10 1 levels 2 region 2 inner 2 2 1" if MUTATE
               else "10 10 1 level 2 inner 2 2 1")
rc_k, e_k, _ = grid(BAD_KEYWORD)
KEY_MSG = "ERROR: Illegal create_grid command (../create_grid.cpp:188)"
no_level_keyword = any(KEY_MSG in e for e in e_k)
print(f"there_is_no_level_keyword={no_level_keyword}")
if not no_level_keyword and not MUTATE:
    print(f"UNEXPECTED: 'level' keyword deck printed {e_k}")

UNSET = ("10 10 1 levels 2 region 2 inner 2 2 1" if MUTATE
         else "10 10 1 levels 2")
rc_u, e_u, _ = grid(UNSET)
UNSET_MSG = "ERROR: Create_grid level was not set (../create_grid.cpp:208)"
unset_caught = any(UNSET_MSG in e for e in e_u)
print(f"every_declared_level_must_be_set={unset_caught}")

rc_z, e_z, _ = grid("10 10 1 levels 2 region 2 inner 2 2 2")
CZ_MSG = ("ERROR: Create_grid cz value must be 1 for a 2d simulation "
          "(../create_grid.cpp:210)")
print(f"the_2d_cz_rule_applies_inside_a_level={any(CZ_MSG in e for e in e_z)}")

# adaptive_grid:7 and :8 — the diagnostics that do not fire.
CIRCLE = find_example("circle", "data.circle")
if CIRCLE is None:
    print("SKIP: SPARTA examples/circle/data.circle not found (set SPARTA_ROOT)")
    sys.exit(0)
AIR = skip_if_unavailable("air.species", "air.vss")

FLOW = """seed 12345
dimension 2
global gridcut 0.0
boundary o r p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
global nrho 1.0 fnum 0.001
species air.species N O
mixture air N O vstream 100.0 0 0
read_surf {circle}
surf_collide W diffuse 300.0 1.0
surf_modify all collide W
collide vss air air.vss
fix in emit/face air xlo
timestep {dt}
fix gc grid/check 1 {mode} {opt}
fix bl balance 100 1.1 rcb cell
stats 100
stats_style step np ncoll nscoll f_gc f_bl
run 300
"""


def flow(dt: float, mode: str = "warn", opt: str = ""):
    rc, txt = run(FLOW.format(circle=CIRCLE, dt=dt, mode=mode, opt=opt), AIR)
    h, r = stats_rows(txt)
    return rc, errors(txt), txt, (h, r)


rc_ok, e_ok, txt_ok, (h_ok, r_ok) = flow(0.0001)
rc_bad, e_bad, txt_bad, (h_bad, r_bad) = flow(0.05, opt="outside yes")

print(f"both_flow_decks_complete={rc_ok == 0 and rc_bad == 0 and not e_ok and not e_bad}")

# The absurd deck is identified as absurd by ITS OWN table, with no stored
# number: every particle strikes the small embedded body more than once per
# timestep, which cannot happen in a physical run. The sane deck is checked to
# be the other side of that line.
absurd = sane = False
if h_ok and h_bad and r_ok and r_bad:
    bad = list(zip(col(h_bad, r_bad, "Np"), col(h_bad, r_bad, "Nscoll")))
    good = list(zip(col(h_ok, r_ok, "Np"), col(h_ok, r_ok, "Nscoll")))
    absurd = any(n > 0 and s > n for n, s in bad)
    sane = all(s <= n for n, s in good if n > 0)
print(f"the_large_timestep_deck_is_physically_absurd={absurd}")
print(f"the_reference_deck_is_not={sane}")

gc_bad = col(h_bad, r_bad, "f_gc") if (h_bad and r_bad) else []
gc_ok = col(h_ok, r_ok, "f_gc") if (h_ok and r_ok) else []
silent = bool(gc_bad) and all(v == 0.0 for v in gc_bad) and all(v == 0.0 for v in gc_ok)
print(f"grid_check_stays_at_zero_on_both={silent}")
print(f"and_prints_no_wrong_cell_warning="
      f"{'particles in wrong cells on timestep' not in txt_bad}")

bl = col(h_bad, r_bad, "f_bl") if (h_bad and r_bad) else []
print(f"fix_balance_reports_exactly_one_on_a_serial_build="
      f"{bool(bl) and all(v == 1.0 for v in bl)}")

rc_m, e_m, _, _ = flow(0.0001, mode="loud")
MODE_MSG = "ERROR: Illegal fix grid/check command (../fix_grid_check.cpp:44)"
print(f"the_mode_word_is_checked={any(MODE_MSG in e for e in e_m)}")

# adaptive_grid:9 — moving surfaces driven into each other fail in the cutter,
# and grid/check does not see it coming.
MOVE = """seed 12345
dimension 2
global gridcut 0.0
boundary o r p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
global nrho 1.0 fnum 0.001
species air.species N O
mixture air N O vstream 100.0 0 0
read_surf {circle} origin 5 5 0 trans 0.0 2.0 0.0 scale 0.33 0.33 1 group g1
read_surf {circle} origin 5 5 0 trans 0.0 -2.0 0.0 scale 0.33 0.33 1 group g2
surf_collide W diffuse 300.0 1.0
surf_modify all collide W
collide vss air air.vss
fix in emit/face air xlo
fix gc grid/check 1 warn
timestep 0.0001
stats 200
stats_style step np nscoll f_gc
run 200
fix m1 move/surf g1 {nev} {nlarge} trans 0 {d} 0
fix m2 move/surf g2 {nev} {nlarge} trans 0 -{d} 0
run 400
"""
rc_gentle, txt_gentle = run(
    MOVE.format(circle=CIRCLE, nev=100, nlarge=2000, d=-0.9), AIR)
print(f"a_gentle_slide_completes={rc_gentle == 0 and not errors(txt_gentle)}")

rc_hard, txt_hard = run(
    MOVE.format(circle=CIRCLE, nev=1, nlarge=2, d=-3.0), AIR)
CUT_MSG = "WB: Point appears last in more than one CLINE (../cut2d.cpp:289)"
cut_fails = any(CUT_MSG in e for e in errors(txt_hard))
print(f"an_aggressive_slide_fails_inside_the_cutter={cut_fails}")
print(f"the_message_names_neither_the_fix_nor_the_step_size="
      f"{cut_fails and 'move/surf' not in ''.join(errors(txt_hard))}")
print(f"a_cut2d_failed_block_precedes_it="
      f"{cut_fails and 'Cut2d failed on proc 0 in cell ID' in txt_hard}")

ok = (rc_r == 0 and rc_s == 0 and rc_3 == 0 and not (e_r + e_s + e_3)
      and no_level_keyword and unset_caught
      and any(CZ_MSG in e for e in e_z)
      and rc_ok == 0 and rc_bad == 0 and absurd and sane and silent
      and "particles in wrong cells on timestep" not in txt_bad
      and bool(bl) and all(v == 1.0 for v in bl)
      and any(MODE_MSG in e for e in e_m)
      and rc_gentle == 0 and cut_fails
      and "Cut2d failed on proc 0 in cell ID" in txt_hard)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
