"""Tier-2: compute thermal/grid across an unresolved velocity gradient — the
effect is real, and it is nothing like the size the entry implies.

  rarefied_flow:5  thermal/grid removes the per-cell MEAN velocity, so across
                   an unresolved gradient the sub-cell shear variance is
                   counted as thermal motion and the temperature reads high;
                   refining drives it down.

A Couette slab: argon between two diffuse walls at the same temperature
translating at -1000 and +1000 m/s, well above the thermal speed. Coarse grid
against a grid ten times finer, three seeds each.

TWO CONFOUNDS HAD TO BE REMOVED BEFORE THIS MEANS ANYTHING, and an earlier
attempt at this claim was abandoned because of the first.

  * rarefied_flow:6 — the instantaneous cell-average of thermal/grid is biased
    LOW when cells hold few particles, and refining the grid lowers occupancy.
    That bias pushes in the SAME direction as the effect under test, so an
    instantaneous reading cannot separate them. Everything here is read through
    'fix ave/grid', which the thermal_grid_is_biased_low_at_few_particles
    fixture establishes removes that bias at every resolution.
  * seed noise — a DSMC cell average is stochastic, so the two grids are each
    run on three seeds and the question asked is whether the two CLUSTERS
    separate, not whether two numbers differ.

WHAT THIS CORRECTS. The entry says the cell-averaged temperature "FALLS
SUBSTANTIALLY when you refine the grid" and offers that as the test. Measured,
with the occupancy bias removed and a strongly supersonic shear, the fall from a
4-cell to a 40-cell grid is about four and a half percent. It is real — the
three coarse values all sit above all three fine values, with no overlap — but
an agent told to look for a substantial fall would not recognise four percent as
the signal, and on a single seed pair it is only a few times the spread. The
magnitude follows from the mechanism: the sub-cell velocity spread is
(dU/dy)*dy, so its variance enters as the square of the cell size and only in
one of three velocity components. The entry now states that.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss")

DECK = """seed {seed}
dimension 2
boundary p s p
global gridcut 0.0
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid {n} {n} 1
global nrho 1e22 fnum 5e11
species ar.species Ar
mixture gas Ar temp 273.15
create_particles gas n 0
collide vss gas ar.vss
surf_collide lo diffuse 273.15 1.0 translate -1000.0 0 0
surf_collide hi diffuse 273.15 1.0 translate 1000.0 0 0
bound_modify ylo collide lo
bound_modify yhi collide hi
compute tg thermal/grid all all temp
fix ftg ave/grid all 1 200 200 c_tg[1]
compute mtf reduce ave f_ftg
compute gt temp
compute npc grid all all n
compute na reduce ave c_npc[1]
timestep 1e-8
stats 400
stats_style step np c_gt c_mtf c_na
run 1600
"""

SEEDS = (12345, 54321, 99999)
COARSE, FINE = 4, 40


def probe(n: int, seed: int) -> dict:
    rc, txt = run(DECK.format(n=n, seed=seed), DATA, timeout=900)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "thermal": column("c_mtf"), "occupancy": column("c_na")}


coarse = [probe(COARSE, s) for s in SEEDS]
fine = [probe(FINE, s) for s in SEEDS]


def last(rs):
    return [r["thermal"][-1] for r in rs if r["thermal"]]


c_vals, f_vals = last(coarse), last(fine)
print(f"coarse_grid_cells_per_side={COARSE} "
      f"occupancy={coarse[0]['occupancy'][-1]:.4g}")
print(f"fine_grid_cells_per_side={FINE} "
      f"occupancy={fine[0]['occupancy'][-1]:.4g}")
print(f"coarse_thermal_grid_by_seed={[round(v, 3) for v in c_vals]}")
print(f"fine_thermal_grid_by_seed={[round(v, 3) for v in f_vals]}")

all_clean = all(r["rc"] == 0 and not r["errors"] and not r["warnings"]
                for r in coarse + fine)
have_all = len(c_vals) == len(SEEDS) == len(f_vals)


def spread(v):
    return (max(v) - min(v)) / (sum(v) / len(v)) if v else 1.0


c_spread, f_spread = spread(c_vals), spread(f_vals)
drop = ((sum(c_vals) / len(c_vals) - sum(f_vals) / len(f_vals))
        / (sum(c_vals) / len(c_vals))) if have_all else 0.0
clusters_separate = have_all and min(c_vals) > max(f_vals)

print(f"coarse_seed_spread={c_spread:.4g}")
print(f"fine_seed_spread={f_spread:.4g}")
print(f"fractional_drop_from_coarse_to_fine={drop:.4g}")

print(f"every_run_exits_zero_with_no_error_and_no_warning={all_clean}")
print(f"all_six_runs_produced_a_reading={have_all}")
print(f"refining_lowers_the_reading_on_every_seed_pairing={clusters_separate}")
print(f"the_effect_is_resolvable_above_the_seed_spread="
      f"{drop > 3.0 * max(c_spread, f_spread)}")
print(f"but_the_drop_is_only_a_few_percent_NOT_substantial={drop < 0.10}")
print(f"the_occupancy_confound_was_removed_by_routing_through_fix_ave_grid="
      f"{coarse[0]['occupancy'][-1] > 10 * fine[0]['occupancy'][-1]}")

ok = (all_clean and have_all and clusters_separate
      and drop > 3.0 * max(c_spread, f_spread) and drop < 0.10
      and coarse[0]["occupancy"][-1] > 10 * fine[0]["occupancy"][-1])
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
