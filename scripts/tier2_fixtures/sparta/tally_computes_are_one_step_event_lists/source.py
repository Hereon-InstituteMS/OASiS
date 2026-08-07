"""Tier-2: the four `*/tally` computes are event lists for ONE timestep, and
`dump tally` is their only consumer.

  chemistry:6  'ITEM: NUMBER OF TALLIES' in a dump tally snapshot equals the
               per-step counter (Nscoll) for that same step, so a dump every N
               steps samples one step in N and is not a total. No fix ave/*
               accepts these computes, stats_style rejects them, dump surf
               rejects them.

The equality between the dump header and the stats column is the whole proof,
and it is a comparison between two outputs of ONE run — SPARTA's own count of
surface collisions on that step against SPARTA's own count of tallies written
for that step. Nothing is compared against a remembered number.

Mutation control: T2_MUTATE=1 swaps ONE token in the three CONSUMER decks — the
compute style word, from 'surf/collision/tally' to 'surf', same group, same
mixture, same deck. A per-surf compute is what those consumers want, so
`dump_surf_rejects_a_tally_compute` and `fix_ave_surf_refuses_a_tally_compute`
both go False. The dump-tally deck is deliberately NOT mutated, so the per-step
evidence is measured on the identical run in both modes.

SCOPE OF THE HOOK, stated plainly. It covers the "only dump tally will take it"
half. The other half — that a snapshot holds ONE timestep's events — cannot be
mutation-controlled from a deck at all, because no edit to any deck makes SPARTA
accumulate them; only a change to SPARTA would. So
`the_dump_header_equals_the_per_step_counter` and
`successive_snapshots_are_not_a_running_total` stay True under T2_MUTATE=1 and
are NOT evidenced by this hook. What backs them instead is that the header is
compared against SPARTA's own Nscoll column for the same step, inside one run.
`stats_style_rejects_a_tally_compute` also stays True, correctly: stats_style
refuses any per-surf compute, tally or not, so the swap cannot flip it.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, find_example, run, run_keep, skip_if_unavailable, stats_rows,
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
read_surf {circle}
surf_collide W diffuse 300.0 0.0
surf_modify all collide W
collide vss air air.vss
fix in emit/face air xlo
timestep 0.0001
compute t {style}
{consumer}
stats 200
stats_style step np nscoll {cols}
run 400
"""

TALLY_STYLE = "surf/collision/tally all all id/surf id type"
# One token differs under mutation, and only in the three CONSUMER decks: the
# compute style word. Group, mixture, deck, seed, geometry and run length are
# identical either way, and the dump-tally deck below is never mutated, so the
# per-step evidence is measured against the same run in both modes.
STYLE = "surf all all n" if MUTATE else TALLY_STYLE
if MUTATE:
    print("mutation=consumer_decks_given_a_plain_compute_surf_instead_of_a_tally_compute")

rc, txt, work = run_keep(
    DECK.format(circle=CIRCLE, style=TALLY_STYLE,
                consumer="dump d tally all 200 dump.tal c_t[*]",
                cols=""),
    DATA)
try:
    h, r = stats_rows(txt)
    steps = col(h, r, "Step") if (h and r) else []
    nscoll = col(h, r, "Nscoll") if (h and r) else []
    per_step = {int(s): n for s, n in zip(steps, nscoll)}

    tallies = {}
    dump = work / "dump.tal"
    if dump.is_file():
        lines = dump.read_text().splitlines()
        step = None
        for i, line in enumerate(lines):
            if line.startswith("ITEM: TIMESTEP"):
                step = int(lines[i + 1].split()[0])
            elif line.startswith("ITEM: NUMBER OF TALLIES") and step is not None:
                tallies[step] = int(lines[i + 1].split()[0])
finally:
    shutil.rmtree(work, ignore_errors=True)

print(f"the_dump_tally_deck_runs={rc == 0 and not errors(txt)}")

# Compare only the steps the stats table reports, EXCLUDING step 0 where both
# are trivially zero and the equality would carry no information.
shared = sorted(s for s in tallies if s in per_step and s > 0)
equal = bool(shared) and all(tallies[s] == per_step[s] for s in shared)
print(f"the_dump_header_equals_the_per_step_counter={equal}")
if not equal:
    print(f"UNEXPECTED: dump steps {sorted(tallies)} vs stats steps "
          f"{sorted(per_step)}; shared={shared}")

# The counter is per-step (universal:8), so a header equal to it is per-step
# too: check the two dumped steps differ, i.e. it is not a running total.
not_cumulative = len(shared) < 2 or not all(
    tallies[b] >= tallies[a] + per_step[a]
    for a, b in zip(shared, shared[1:]))
print(f"successive_snapshots_are_not_a_running_total={not_cumulative}")

# Nothing else will take these computes.
rc_s, txt_s = run(DECK.format(circle=CIRCLE, style=STYLE, consumer="",
                              cols="c_t"), DATA)
STATS_MSG = "ERROR: Stats compute does not compute scalar (../stats.cpp:678)"
print(f"stats_style_rejects_a_tally_compute="
      f"{any(STATS_MSG in e for e in errors(txt_s))}")

rc_d, txt_d = run(
    DECK.format(circle=CIRCLE, style=STYLE,
                consumer="dump d surf all 200 dump.srf id c_t[*]", cols=""),
    DATA)
DUMPSURF_MSG = ("ERROR: Dump surf compute does not compute per-surf info "
                "(../dump_surf.cpp:567)")
print(f"dump_surf_rejects_a_tally_compute="
      f"{any(DUMPSURF_MSG in e for e in errors(txt_d))}")

rc_f, txt_f = run(
    DECK.format(circle=CIRCLE, style=STYLE,
                consumer="fix fa ave/surf all 10 20 200 c_t[1]", cols=""),
    DATA)
avesurf_refuses = rc_f != 0 and bool(errors(txt_f))
print(f"fix_ave_surf_refuses_a_tally_compute={avesurf_refuses}")
if avesurf_refuses:
    print(f"ave_surf_message={errors(txt_f)[0]}")

ok = (rc == 0 and not errors(txt) and equal and not_cumulative
      and any(STATS_MSG in e for e in errors(txt_s))
      and any(DUMPSURF_MSG in e for e in errors(txt_d))
      and avesurf_refuses)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
