"""Tier-2: compute thermal/grid reads far too cold when cells hold few
particles, and routing it through a fix ave/grid puts it back.

  rarefied_flow:6  thermal/grid subtracts the per-cell SAMPLE mean velocity,
                   which costs a degree of freedom, and SPARTA writes exactly
                   0 K into any cell holding one particle or none. A
                   'compute reduce ave' then averages those hard zeros in.

Three decks. Same gas, same temperature, same fnum, same everything except the
grid resolution — so the particles per cell fall while the physics does not
change. The instantaneous cell-averaged thermal/grid temperature is compared, in
the SAME run, against 'compute temp' (which is a global estimator and is not
biased this way) and against the same quantity routed through a fix ave/grid.
Nothing is asserted against a stored temperature: the assertions are that the
instantaneous estimator falls away from the global one as the cells empty, that
the global one does not move, and that the time-averaged route tracks the global
one at every resolution.
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
compute tg thermal/grid all all temp
compute mt reduce ave c_tg[1]
fix ftg ave/grid all 1 100 100 c_tg[1]
compute mtf reduce ave f_ftg
compute gt temp
compute npc grid all all n
compute na reduce ave c_npc[1]
timestep 1e-7
stats 100
stats_style step np c_gt c_mt c_mtf c_na
run 300
"""


def probe(n: int) -> dict:
    rc, txt = run(DECK.format(n=n), DATA)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "global_temp": column("c_gt"), "cell_ave": column("c_mt"),
            "fix_ave": column("c_mtf"), "per_cell": column("c_na")}


many = probe(20)         # tens of particles per cell
few = probe(100)         # a couple per cell
almost_none = probe(200)  # under one per cell

for tag, r in (("tens_per_cell", many), ("a_couple_per_cell", few),
               ("under_one_per_cell", almost_none)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")
    print(f"{tag}_particles_per_cell={r['per_cell'][-1]:.4g} "
          f"global_temp={r['global_temp'][-1]:.4g} "
          f"instantaneous_cell_average={r['cell_ave'][-1]:.4g} "
          f"time_averaged={r['fix_ave'][-1]:.4g}")


def frac(r, key):
    g = r["global_temp"][-1]
    return (r[key][-1] / g) if g else 0.0


all_clean = all(r["rc"] == 0 and not r["errors"] and not r["warnings"]
                for r in (many, few, almost_none))
occupancy_falls = (many["per_cell"][-1] > few["per_cell"][-1]
                   > almost_none["per_cell"][-1])
global_temp_is_stable = (max(r["global_temp"][-1]
                             for r in (many, few, almost_none))
                         / min(r["global_temp"][-1]
                               for r in (many, few, almost_none))) < 1.05
instantaneous_falls = (frac(many, "cell_ave") > frac(few, "cell_ave")
                       > frac(almost_none, "cell_ave"))
few_reads_about_a_third = frac(few, "cell_ave") < 0.5
almost_none_reads_far_worse = frac(almost_none, "cell_ave") < 0.15
tens_per_cell_is_nearly_right = frac(many, "cell_ave") > 0.9
fix_average_recovers = all(frac(r, "fix_ave") > 0.9
                           for r in (many, few, almost_none))

print(f"every_run_exits_zero_with_no_error_and_no_warning={all_clean}")
print(f"particles_per_cell_fall_as_the_grid_refines={occupancy_falls}")
print(f"the_global_compute_temp_does_not_move={global_temp_is_stable}")
print(f"the_instantaneous_cell_average_falls_with_occupancy="
      f"{instantaneous_falls}")
print(f"at_tens_of_particles_per_cell_it_is_nearly_right="
      f"{tens_per_cell_is_nearly_right}")
print(f"at_a_couple_per_cell_it_reads_under_half_the_true_temperature="
      f"{few_reads_about_a_third}")
print(f"under_one_per_cell_it_collapses_further={almost_none_reads_far_worse}")
print(f"routing_through_fix_ave_grid_recovers_it_at_every_resolution="
      f"{fix_average_recovers}")

ok = (all_clean and occupancy_falls and global_temp_is_stable
      and instantaneous_falls and tens_per_cell_is_nearly_right
      and few_reads_about_a_third and almost_none_reads_far_worse
      and fix_average_recovers)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
