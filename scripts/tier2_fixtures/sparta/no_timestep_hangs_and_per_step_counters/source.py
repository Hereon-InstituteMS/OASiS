"""Tier-2 for the four cross-cutting SPARTA claims nothing was defending.

  universal:5  there is no usable default timestep: dt starts at 1.0 SECOND, so
               a deck that omits 'timestep' on a small domain does not fail, it
               hangs, and it hangs with a ZERO-BYTE log.
  universal:6  a seed change moves every tallied number; the same seed
               reproduces them exactly; and Ncollave settles far faster than
               Ncoll.
  universal:7  in 2d the cell volume is per ONE METRE of depth and the z extent
               handed to create_box is ignored for volume.
  universal:8  Ncoll and its siblings are the count on THAT ONE TIMESTEP, not a
               total and not a sum over the stats interval.

All four are read off decks that differ in exactly one line, and none of them is
asserted against a stored number.

ON THE ONE TIMING-SHAPED ASSERTION. universal:5 is a claim about a hang, so it
cannot be checked without a clock somewhere. What is ASSERTED is structural —
the run never emits its first stats line and its captured output is empty —
while the clock only bounds how long the fixture waits. That asymmetry is what
makes it safe on a loaded host: a slower machine makes the no-timestep deck
MORE certain to be still running, never less. The control deck, identical but
for one 'timestep' line, completes and prints its table.

Mutation control: T2_MUTATE=1 gives the deck that OMITTED 'timestep' the same
'timestep 1e-7' line the control deck carries, i.e. it removes the
no-default-timestep pathology and nothing else. That deck then returns rc=0
within the wait, prints its stats table, so `omitting_timestep_never_returns`
and `omitting_timestep_prints_nothing_at_all` go False and
`no_timestep_reached_a_stats_line` goes True. The universal:6/:7/:8 decks are
untouched, so those expectations are unaffected — this control covers
universal:5 only.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, run, skip_if_unavailable, stats_rows,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

# A 1 mm box. Everything below differs from this by one line.
DECK = """seed {seed}
dimension 2
global gridcut 0.0
boundary p p p
create_box 0 1e-3 0 1e-3 {z}
create_grid 20 20 1
global nrho 1e22 fnum 5e11
species ar.species Ar
mixture gas Ar temp 273.15
create_particles gas n 0
collide vss gas ar.vss
{dt}stats {every}
stats_style step np ncoll ncollave
run 300
"""


def deck(seed=12345, z="-0.5 0.5", dt="timestep 1e-7\n", every=100):
    return DECK.format(seed=seed, z=z, dt=dt, every=every)


def probe(text: str, timeout: int = 600) -> dict:
    rc, txt = run(text, DATA, timeout=timeout)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    created = re.search(r"Created (\d+) particles", txt)
    return {"rc": rc, "errors": errors(txt), "output": txt,
            "has_stats_header": bool(header),
            "created_line": created.group(0) if created else "",
            "ncoll": column("Ncoll"), "ncollave": column("Ncollave"),
            "np": column("Np")}


# ------------------------------------------------------------- universal:5
# 25 s is not the claim; it is only how long this fixture is willing to wait.
HANG_WAIT = 25
if MUTATE:
    print("mutation=no_timestep_deck_given_the_same_timestep_1e-7_line")
# Under mutation the pathology is REMOVED: the deck that omitted 'timestep' is
# handed the control deck's 'timestep 1e-7' line, and nothing else changes.
no_dt = probe(deck(dt="timestep 1e-7\n" if MUTATE else ""), timeout=HANG_WAIT)
with_dt = probe(deck())

print(f"no_timestep_returncode={no_dt['rc']}")
print(f"no_timestep_captured_output_bytes={len(no_dt['output'])}")
print(f"no_timestep_reached_a_stats_line={no_dt['has_stats_header']}")
print(f"with_timestep_reached_a_stats_line={with_dt['has_stats_header']}")
print(f"with_timestep_returncode={with_dt['rc']}")
# _spahelp.run reports -99 when the deck outlives the timeout.
print(f"omitting_timestep_never_returns={no_dt['rc'] == -99}")
print(f"omitting_timestep_prints_nothing_at_all="
      f"{no_dt['output'].strip() == ''}")
print(f"the_only_difference_is_one_timestep_line="
      f"{with_dt['rc'] == 0 and with_dt['has_stats_header']}")

# ------------------------------------------------------------- universal:7
thick = probe(deck(dt="timestep 1e-7\n", z="-0.5 0.5"))
thin = probe(deck(dt="timestep 1e-7\n", z="-0.0005 0.0005"))
print(f"z_extent_1m_created={thick['created_line']}")
print(f"z_extent_1mm_created={thin['created_line']}")
print(f"a_1000x_thinner_z_extent_creates_the_same_count="
      f"{bool(thick['created_line']) and thick['created_line'] == thin['created_line']}")

# ------------------------------------------------------------- universal:6
same_a = with_dt
same_b = probe(deck())
other = probe(deck(seed=54321))

identical = (bool(same_a["ncoll"]) and same_a["ncoll"] == same_b["ncoll"]
             and same_a["ncollave"] == same_b["ncollave"])
differs = (bool(other["ncoll"]) and other["ncoll"][-1] != same_a["ncoll"][-1])


def rel(a, b):
    return abs(a - b) / a if a else 0.0


ncoll_spread = rel(same_a["ncoll"][-1], other["ncoll"][-1]) if differs else 0.0
ncollave_spread = (rel(same_a["ncollave"][-1], other["ncollave"][-1])
                   if same_a["ncollave"] and other["ncollave"] else 0.0)
print(f"same_seed_reproduces_every_column_exactly={identical}")
print(f"changing_only_the_seed_changes_the_tally={differs}")
print(f"running_mean_is_tighter_across_seeds_than_the_per_step_count="
      f"{ncollave_spread < ncoll_spread}")

# ------------------------------------------------------------- universal:8
every_step = probe(deck(every=1))
# Same seed, same deck, same final step: if Ncoll were a sum over the stats
# interval the two would differ by ~the interval. Being EQUAL is what proves it
# is the count on one timestep.
per_step_equal = (bool(every_step["ncoll"]) and bool(same_a["ncoll"])
                  and every_step["ncoll"][-1] == same_a["ncoll"][-1])
print(f"stats_1_and_stats_100_report_the_same_final_Ncoll={per_step_equal}")
print(f"the_column_is_a_per_step_count_not_an_interval_sum={per_step_equal}")
print(f"running_mean_also_matches_across_stats_intervals="
      f"{bool(every_step['ncollave']) and every_step['ncollave'][-1] == same_a['ncollave'][-1]}")

ok = (no_dt["rc"] == -99 and not no_dt["has_stats_header"]
      and no_dt["output"].strip() == ""
      and with_dt["rc"] == 0 and with_dt["has_stats_header"]
      and thick["created_line"] and thick["created_line"] == thin["created_line"]
      and identical and differs and ncollave_spread < ncoll_spread
      and per_step_equal)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
