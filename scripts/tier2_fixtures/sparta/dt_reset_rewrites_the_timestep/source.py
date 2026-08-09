"""Tier-2: `fix dt/reset` can throw away the timestep you set, and one
argument decides whether it does.

  collision_relaxation:8  resetflag 0 only reports a recommendation through f_ID;
                    1 or 2 WRITE it into the global timestep, so every result
                    after the first Nevery steps belongs to a step size the
                    deck did not choose. The only place it shows is the 'dt'
                    stats column.

Two decks, identical apart from the last argument of the fix. Both publish the
same recommendation through f_ID — that is what makes the comparison clean: the
fix is doing the same arithmetic in both, and only the write-back differs.

WHAT IS ASSERTED: that the Dt column is CONSTANT in one deck and CHANGES in the
other, and that where it changes it equals the fix's own published
recommendation. Both are relations inside a single run. No timestep value, no
ratio and no tolerance is remembered here.

Mutation control: T2_MUTATE=1 sets the resetflag of the rewriting deck to 0, the
single edit that removes the pathology, leaving the compute chain, the weight,
Nevery, the seed and the run length alone. The two decks become the same deck,
the Dt column stops moving, and `resetflag_one_rewrites_the_global_timestep` and
`the_new_dt_is_the_fixes_own_recommendation` go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

DECK = """seed 12345
dimension 2
boundary p p p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e11
create_particles gas n 0
collide vss gas ar.vss
timestep 1e-8
compute g grid all all nrho temp usq vsq wsq
fix fg ave/grid all 10 10 100 c_g[*]
compute lam lambda/grid f_fg[1] f_fg[2] lambda tau
compute dtg dt/grid all 0.25 0.25 c_lam[2] f_fg[2] f_fg[3] f_fg[4] f_fg[5]
fix dr dt/reset 100 c_dtg 0.1 {resetflag}
stats 100
stats_style step np dt f_dr
run 300
"""


def go(resetflag: int):
    rc, txt = run(DECK.format(resetflag=resetflag), DATA)
    h, r = stats_rows(txt)
    if rc or not r:
        return rc, errors(txt), [], []
    return rc, errors(txt), col(h, r, "Dt"), col(h, r, "f_dr")


if MUTATE:
    print("mutation=resetflag_of_the_rewriting_deck_set_to_zero")

rc0, e0, dt0, rec0 = go(0)
rc1, e1, dt1, rec1 = go(0 if MUTATE else 1)

both_ran = rc0 == 0 and rc1 == 0 and not e0 and not e1 and len(dt0) > 2
print(f"both_decks_complete={both_ran}")

report_only_holds_dt = bool(dt0) and all(v == dt0[0] for v in dt0)
print(f"resetflag_zero_leaves_the_timestep_alone={report_only_holds_dt}")

# The report-only deck still publishes a recommendation, and it differs from the
# timestep the deck set — otherwise the two decks could not be told apart.
publishes_a_recommendation = (len(rec0) > 1 and rec0[0] == 0.0
                              and any(v != 0.0 and v != dt0[0] for v in rec0[1:]))
print(f"resetflag_zero_still_publishes_a_recommendation={publishes_a_recommendation}")

rewrites = bool(dt1) and any(v != dt1[0] for v in dt1)
print(f"resetflag_one_rewrites_the_global_timestep={rewrites}")
if not rewrites and not MUTATE:
    print(f"UNEXPECTED: resetflag 1 left the Dt column constant at {dt1[:1]}")

# Where it changed, the new Dt is the fix's own published value on that row.
matches = False
if rewrites and len(dt1) == len(rec1):
    matches = all(d == r for d, r in zip(dt1[1:], rec1[1:]))
print(f"the_new_dt_is_the_fixes_own_recommendation={matches}")

ok = (both_ran and report_only_holds_dt and publishes_a_recommendation
      and rewrites and matches)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
