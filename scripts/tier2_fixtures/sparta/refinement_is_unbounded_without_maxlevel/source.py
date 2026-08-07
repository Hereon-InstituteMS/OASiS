"""Tier-2: 'fix adapt' refines without limit unless you cap it with 'maxlevel'.

  adaptive_grid:2    every level multiplies the cell count and divides the
                     particles per cell; below one particle per cell every
                     per-cell diagnostic, including the one driving the
                     adaptation, is noise.
  hypersonic_flow:5  the same claim stated for the hypersonic row.

Two decks. They differ by the four characters 'maxlevel 2' at the end of one
'fix adapt' line, and nothing else. Every assertion is a shape — a column that
is flat, a column that climbs, a column that saturates — or a comparison between
the two decks. No count from either run is pinned, which matters because the
cell counts a refinement produces depend on the seed and on the particle
distribution.

Mutation control: T2_MUTATE=1 gives the "uncapped" deck the same `maxlevel 2`
cap as the control, i.e. it removes the unbounded-refinement pathology and
nothing else. `uncapped_maxlevel_exceeds_the_cap`,
`uncapped_ngrid_climbs_every_stats_line`, `per_cell_minimum_falls_below_one`,
`per_cell_average_more_than_halves` and
`uncapped_ends_with_more_cells_than_capped` all go False.
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
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15
global nrho 7.07043e22 fnum 7.07043e11
collide vss gas ar.vss
create_particles gas n 0
compute npc grid all all n
compute mn reduce min c_npc[1]
compute av reduce ave c_npc[1]
timestep 1e-9
stats 100
stats_style step np ngrid maxlevel c_mn c_av
fix ad adapt 100 all refine particle 5 0 {cap}
run 600
"""


def probe(cap: str) -> dict:
    rc, txt = run(DECK.format(cap=cap), DATA)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "np": column("Np"), "ngrid": column("Ngrid"),
            "maxlevel": column("Maxlevel"), "cell_min": column("c_mn"),
            "cell_ave": column("c_av")}


# Under mutation the pathology is REMOVED: the "uncapped" deck is given the same
# maxlevel cap as the control, so the refinement is bounded in both runs.
uncapped = probe("maxlevel 2" if MUTATE else "")
capped = probe("maxlevel 2")
if MUTATE:
    print("mutation=uncapped_deck_given_the_same_maxlevel_2_cap")

for tag, r in (("uncapped", uncapped), ("capped_at_level_2", capped)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")
    print(f"{tag}_ngrid_column={[int(v) for v in r['ngrid']]}")
    print(f"{tag}_maxlevel_column={[int(v) for v in r['maxlevel']]}")
    print(f"{tag}_np_column={[int(v) for v in r['np']]}")


def strictly_climbs(c):
    return len(c) > 2 and all(b > a for a, b in zip(c, c[1:]))


def flat(c):
    return len(c) > 2 and len(set(c)) == 1


def saturates(c):
    return len(c) > 3 and c[-1] == c[-2] and c[-1] > c[0]


both_zero = uncapped["rc"] == 0 and capped["rc"] == 0
silent = not uncapped["warnings"] and not capped["warnings"]
level_exceeds = bool(uncapped["maxlevel"]) and uncapped["maxlevel"][-1] > 2
min_falls = (bool(uncapped["cell_min"]) and uncapped["cell_min"][0] >= 1
             and uncapped["cell_min"][-1] < 1)
ave_halves = (bool(uncapped["cell_ave"])
              and uncapped["cell_ave"][-1] < 0.5 * uncapped["cell_ave"][0])
level_capped = bool(capped["maxlevel"]) and max(capped["maxlevel"]) <= 2
more_cells = (bool(uncapped["ngrid"]) and bool(capped["ngrid"])
              and uncapped["ngrid"][-1] > capped["ngrid"][-1])

print(f"both_runs_exit_zero={both_zero}")
print(f"neither_run_warns={silent}")
print(f"uncapped_ngrid_climbs_every_stats_line={strictly_climbs(uncapped['ngrid'])}")
print(f"uncapped_maxlevel_exceeds_the_cap={level_exceeds}")
print(f"particle_count_is_flat_while_cells_multiply={flat(uncapped['np'])}")
print(f"per_cell_minimum_falls_below_one={min_falls}")
print(f"per_cell_average_more_than_halves={ave_halves}")
print(f"maxlevel_caps_the_level={level_capped}")
print(f"maxlevel_makes_the_cell_count_saturate={saturates(capped['ngrid'])}")
print(f"uncapped_ends_with_more_cells_than_capped={more_cells}")

ok = (uncapped["rc"] == 0 and capped["rc"] == 0
      and not uncapped["warnings"] and not capped["warnings"]
      and strictly_climbs(uncapped["ngrid"])
      and uncapped["maxlevel"] and uncapped["maxlevel"][-1] > 2
      and flat(uncapped["np"])
      and uncapped["cell_min"][0] >= 1 and uncapped["cell_min"][-1] < 1
      and uncapped["cell_ave"][-1] < 0.5 * uncapped["cell_ave"][0]
      and max(capped["maxlevel"]) <= 2
      and saturates(capped["ngrid"])
      and uncapped["ngrid"][-1] > capped["ngrid"][-1])
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
