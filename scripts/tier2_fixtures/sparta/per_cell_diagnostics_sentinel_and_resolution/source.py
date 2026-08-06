"""Tier-2: the four ways a SPARTA per-cell diagnostic lies to you while the run
exits 0.

  rarefied_flow:1  compute lambda/grid reading a fix ave/grid that has produced
                   no output returns the BIG sentinel, and on a grid refined
                   past one particle per cell it keeps returning it forever in
                   the cells that are empty — the MAXIMUM stays pinned while
                   the minimum looks perfectly healthy.
  rarefied_flow:3  an fnum (or a refinement) leaving under one particle per
                   cell runs, reports a plausible collision rate, and turns
                   every per-cell quantity into garbage.
  rarefied_flow:0  cells larger than the local mean free path run cleanly and
                   a particle-count check does NOT catch it, because a coarse
                   grid has MORE particles per cell.
  rarefied_flow:2  a timestep far larger than the mean collision time runs
                   cleanly, and compute dt/grid reports the same recommendation
                   whatever timestep you set — SPARTA never compares the two.
  rarefied_flow:7  and none of that shows up in the collision column, because a
                   box at rest has no gradient for the errors to act on: the
                   collision RATE is the same across a ten-thousand-fold
                   timestep change and across a sixteen-fold change in cell
                   count. A dt or grid study on an equilibrium box measures
                   nothing.

Six decks, all the same 1 mm 2d argon box, differing only in the grid
resolution, the fix ave/grid sample count, or the timestep. Nothing here is
asserted against a stored value: the sentinel is identified by being fifteen
orders of magnitude away from the healthy value IN THE SAME RUN, the resolution
claims are comparisons between two grids, and the dt/grid claim is the ratio of
two recommendations against the ratio of the two timesteps that produced them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss")

DECK = """seed 12345
dimension 2
boundary p p p
global gridcut 0.0
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid {n} {n} 1
global nrho 1e22 fnum 5e11
species ar.species Ar
mixture gas Ar temp 273.15
create_particles gas n 0
collide vss gas ar.vss
compute g grid all all nrho temp usq vsq wsq
fix fg ave/grid all 1 {nrepeat} 50 c_g[*]
compute lam lambda/grid f_fg[1] f_fg[2] lambda knall
compute mct lambda/grid f_fg[1] f_fg[2] tau
compute dtg dt/grid all 0.25 0.1 c_mct f_fg[2] f_fg[3] f_fg[4] f_fg[5]
compute npc grid all all n
compute knmin reduce min c_lam[2]
compute knmax reduce max c_lam[2]
compute lmax reduce max c_lam[1]
compute nmin reduce min c_npc[1]
compute dtmin reduce min c_dtg
timestep {dt}
stats 50
stats_style step np ncoll c_nmin c_lmax c_knmin c_knmax c_dtmin
run 200
"""


def probe(n=20, nrepeat=50, dt="1e-7") -> dict:
    rc, txt = run(DECK.format(n=n, nrepeat=nrepeat, dt=dt), DATA)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "ncoll": column("Ncoll"), "nmin": column("c_nmin"),
            "lmax": column("c_lmax"), "knmin": column("c_knmin"),
            "knmax": column("c_knmax"), "dtmin": column("c_dtmin")}


coarse = probe(n=5)                      # cells LARGER than a mean free path
resolved = probe(n=20)                   # the reference resolution
refined = probe(n=200, nrepeat=1)        # far past one particle per cell
one_sample = probe(n=100, nrepeat=1)     # empty cells survive the averaging
big_dt = probe(n=20, dt="1e-5")          # 100x the timestep
small_dt = probe(n=20, dt="1e-9")        # 1/100 the timestep

for tag, r in (("coarse_5x5", coarse), ("resolved_20x20", resolved),
               ("refined_200x200", refined), ("one_sample_100x100", one_sample),
               ("timestep_100x_larger", big_dt),
               ("timestep_100x_smaller", small_dt)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")

print(f"unfed_fix_step0_lambda_max={resolved['lmax'][0]:.4g}")
print(f"unfed_fix_step0_kn_min={resolved['knmin'][0]:.4g}")
print(f"unfed_fix_step0_kn_max={resolved['knmax'][0]:.4g}")
print(f"settled_kn_min={resolved['knmin'][-1]:.4g}")
print(f"coarse_kn_min={coarse['knmin'][-1]:.4g} "
      f"coarse_cell_particle_min={coarse['nmin'][-1]:.4g}")
print(f"resolved_kn_min={resolved['knmin'][-1]:.4g} "
      f"resolved_cell_particle_min={resolved['nmin'][-1]:.4g}")
print(f"refined_cell_particle_min={refined['nmin'][-1]:.4g}")
print(f"refined_kn_min={refined['knmin'][-1]:.4g} "
      f"refined_kn_max={refined['knmax'][-1]:.4g}")
print(f"one_sample_kn_min={one_sample['knmin'][-1]:.4g} "
      f"one_sample_kn_max={one_sample['knmax'][-1]:.4g}")
print(f"recommended_dt_at_100x_larger_timestep={big_dt['dtmin'][-1]:.4g}")
print(f"recommended_dt_at_100x_smaller_timestep={small_dt['dtmin'][-1]:.4g}")

SENTINEL_GAP = 1e15          # the sentinel is not "large", it is unphysical

all_clean = all(r["rc"] == 0 and not r["errors"] and not r["warnings"]
                for r in (coarse, resolved, refined, one_sample, big_dt,
                          small_dt))
step0_is_sentinel = (resolved["knmin"][0] > SENTINEL_GAP * resolved["knmin"][-1]
                     and resolved["lmax"][0] > SENTINEL_GAP * resolved["lmax"][-1])
sentinel_persists = (one_sample["knmax"][-1]
                     > SENTINEL_GAP * one_sample["knmin"][-1]
                     and refined["knmax"][-1]
                     > SENTINEL_GAP * refined["knmin"][-1])
minimum_still_looks_healthy = (one_sample["knmin"][-1] < 1e3
                               and refined["knmin"][-1] < 1e3)
empty_cells_on_the_refined_grid = (refined["nmin"][-1] == 0
                                   and one_sample["nmin"][-1] == 0)
coarse_cells_exceed_a_mean_free_path = coarse["knmin"][-1] < 1.0
resolved_cells_do_not = resolved["knmin"][-1] > 1.0
coarse_has_MORE_particles_per_cell = coarse["nmin"][-1] > resolved["nmin"][-1]
dt_ratio = (big_dt["dtmin"][-1] / small_dt["dtmin"][-1]
            if small_dt["dtmin"][-1] else 0.0)
recommendation_ignores_your_timestep = 0.5 < dt_ratio < 2.0
too_large_a_timestep_is_accepted = (big_dt["rc"] == 0
                                    and big_dt["dtmin"][-1] < 1e-5)

print(f"every_run_exits_zero_with_no_error_and_no_warning={all_clean}")
print(f"unfed_fix_returns_the_sentinel_at_step_0={step0_is_sentinel}")
print(f"the_sentinel_persists_on_a_refined_grid={sentinel_persists}")
print(f"while_the_minimum_still_looks_healthy={minimum_still_looks_healthy}")
print(f"the_refined_grid_has_empty_cells={empty_cells_on_the_refined_grid}")
print(f"coarse_cells_are_larger_than_a_mean_free_path="
      f"{coarse_cells_exceed_a_mean_free_path}")
print(f"the_resolved_grid_is_not={resolved_cells_do_not}")
print(f"a_particle_count_check_does_NOT_catch_the_coarse_grid="
      f"{coarse_has_MORE_particles_per_cell}")
print(f"dt_grid_recommends_the_same_dt_across_a_10000x_timestep_change="
      f"{recommendation_ignores_your_timestep}")
print(f"a_timestep_far_above_the_recommendation_still_exits_zero="
      f"{too_large_a_timestep_is_accepted}")

# ----------------------------------------------------------- rarefied_flow:7
# Why none of the above was caught by looking at the collision column. Ncoll is
# a per-step count, so it scales with dt trivially; the physical quantity is the
# collision RATE. In a box at rest that rate is the same whether the timestep is
# a hundred times too large or a hundred times too small, and the same on a grid
# with sixteen times fewer cells. A dt or grid study run on an equilibrium box
# is therefore measuring nothing.
def rate(r, dt):
    return (r["ncoll"][-1] / dt) if r["ncoll"] else 0.0


rate_big = rate(big_dt, 1e-5)
rate_ref = rate(resolved, 1e-7)
rate_small = rate(small_dt, 1e-9)
rate_coarse = rate(coarse, 1e-7)


def spread(*values):
    lo, hi = min(values), max(values)
    return (hi - lo) / hi if hi else 1.0


def ratio(*values):
    lo, hi = min(values), max(values)
    return (hi / lo) if lo else float("inf")


timestep_rate_spread = spread(rate_big, rate_ref, rate_small)
grid_rate_spread = spread(rate_ref, rate_coarse)
count_ratio = ratio(big_dt["ncoll"][-1], resolved["ncoll"][-1],
                    small_dt["ncoll"][-1])
rate_ratio = ratio(rate_big, rate_ref, rate_small)
count_moves_more = count_ratio > 100 * rate_ratio

print(f"collision_rate_spread_over_a_10000x_timestep_change="
      f"{timestep_rate_spread:.4g}")
print(f"collision_rate_spread_over_a_16x_cell_count_change="
      f"{grid_rate_spread:.4g}")
print(f"raw_Ncoll_max_over_min={count_ratio:.4g} "
      f"collision_rate_max_over_min={rate_ratio:.4g}")
print(f"the_raw_Ncoll_column_moves_far_more_than_the_rate={count_moves_more}")
equilibrium_box_measures_nothing = (timestep_rate_spread < 0.2
                                    and grid_rate_spread < 0.2)
print(f"an_equilibrium_box_is_insensitive_to_both_errors="
      f"{equilibrium_box_measures_nothing}")

ok = (all_clean and step0_is_sentinel and sentinel_persists
      and minimum_still_looks_healthy and empty_cells_on_the_refined_grid
      and coarse_cells_exceed_a_mean_free_path and resolved_cells_do_not
      and coarse_has_MORE_particles_per_cell
      and recommendation_ignores_your_timestep
      and too_large_a_timestep_is_accepted
      and equilibrium_box_measures_nothing
      and count_moves_more)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
