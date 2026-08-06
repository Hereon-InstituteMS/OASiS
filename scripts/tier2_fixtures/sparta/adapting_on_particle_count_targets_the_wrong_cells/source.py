"""Tier-2: a uniform grid sized on freestream conditions is far too coarse at
the shock, and refining on particle count does not fix it.

  hypersonic_flow:3  the run is clean and the shock is simply smeared; the cell
                     Knudsen number from 'compute lambda/grid ... knall' with
                     'compute reduce min' is below 1 somewhere in the domain
                     and SPARTA never warns.
  adaptive_grid:4    adapting on particle count refines where particles already
                     are, not where the gradients are, so the cell Knudsen
                     minimum does not improve; adapting on the Knudsen column
                     targets the under-resolution directly.

A physically scaled Mach-5 argon flow past a cylinder — scaled deliberately,
because upstream's examples/circle uses nrho 1.0, which makes the flow free
molecular (cell Knudsen numbers around 1e18) and leaves nothing to resolve. Here
the freestream mean free path is a fraction of a cell and the cell Knudsen
minimum sits far below 1.

Three decks, identical but for the 'fix adapt' line.

WHAT THIS SHARPENS. adaptive_grid:4's Signal bundles two things: "the cell
Knudsen minimum has not improved EVEN THOUGH Ngrid has grown several-fold". The
first half holds and the second does not. Refining on particle count splits a
cell's particles among its children, which drops them below the threshold
immediately, so the criterion is self-limiting: measured, the cell count grows
by about a quarter, not several-fold, while the Knudsen-driven run on the same
deck grows more than fivefold. An agent looking for several-fold growth AND no
improvement would not find the first half and could conclude the entry did not
apply to its case. The observable that matters is just that the Knudsen minimum
does not move.

Both ratios were measured on two seeds before any threshold here was chosen:
particle-count gave 0.947 and 0.928 (no improvement), Knudsen-driven gave 3.19
and 3.08. The gates below sit far from both clusters, which is how a Monte-Carlo
quantity has to be bounded.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, find_example, run, skip_if_unavailable, stats_rows,
)

DATA = skip_if_unavailable("ar.species", "ar.vss")
CIRCLE = find_example("circle", "data.circle")
if CIRCLE is None:                                          # pragma: no cover
    print("SKIP: SPARTA examples/circle/data.circle not found (set SPARTA_ROOT)")
    sys.exit(0)
FILES = dict(DATA, **{"data.circle": CIRCLE})

DECK = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o o p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
global nrho 1e19 fnum 2e16
species ar.species Ar
mixture gas Ar vstream 1600.0 0 0 temp 300.0
read_surf data.circle
surf_collide cw diffuse 300.0 1.0
surf_modify all collide cw
collide vss gas ar.vss
create_particles gas n 0
fix in emit/face gas xlo
compute g grid all all nrho temp
fix fg ave/grid all 1 50 50 c_g[*]
compute lam lambda/grid f_fg[1] f_fg[2] lambda knall
compute knmin reduce min c_lam[2]
compute npc grid all all n
compute nmax reduce max c_npc[1]
timestep 1e-4
stats 800
stats_style step np ngrid c_knmin c_nmax
run 800
{adapt}run 800
"""

PARTICLE = "fix ad adapt 200 all refine particle 500 0 maxlevel 3\n"
KNUDSEN = ("fix ad adapt 200 all refine value c_lam[2] 0.5 1e20 "
           "thresh less more maxlevel 3\n")


def probe(adapt: str) -> dict:
    rc, txt = run(DECK.format(adapt=adapt), FILES, timeout=1500)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "knmin": column("c_knmin"), "ngrid": column("Ngrid"),
            "np": column("Np")}


baseline = probe("")
particle = probe(PARTICLE)
knudsen = probe(KNUDSEN)

for tag, r in (("no_adaptation", baseline),
               ("adapt_on_particle_count", particle),
               ("adapt_on_cell_knudsen", knudsen)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")
    print(f"{tag}_ngrid={[int(v) for v in r['ngrid']]} "
          f"knmin={[round(v, 6) for v in r['knmin']]}")

# The pre-adaptation state is the FIRST stats row of every deck: all three are
# the identical run up to that point, so it is the common reference.
ref_kn = baseline["knmin"][0] if baseline["knmin"] else 0.0
ref_ngrid = baseline["ngrid"][0] if baseline["ngrid"] else 0.0


def gain(r):
    return (r["knmin"][-1] / ref_kn) if ref_kn else 0.0


def growth(r):
    return (r["ngrid"][-1] / ref_ngrid) if ref_ngrid else 0.0


print(f"cell_knudsen_minimum_before_adaptation={ref_kn:.6g}")
print(f"particle_driven_knudsen_gain={gain(particle):.4g} "
      f"cell_growth={growth(particle):.4g}")
print(f"knudsen_driven_knudsen_gain={gain(knudsen):.4g} "
      f"cell_growth={growth(knudsen):.4g}")

all_clean = all(r["rc"] == 0 and not r["errors"] and not r["warnings"]
                for r in (baseline, particle, knudsen))
# hypersonic_flow:3 — under-resolved, silently.
under_resolved = 0.0 < ref_kn < 1.0
baseline_never_warns = not baseline["warnings"] and baseline["rc"] == 0
# adaptive_grid:4 — the observable that matters.
particle_does_not_improve_it = gain(particle) < 1.1
knudsen_does_improve_it = gain(knudsen) > 2.0
# ...and the half of the published Signal that does NOT hold.
particle_criterion_is_self_limiting = (growth(particle) < 2.0
                                       and growth(knudsen)
                                       > 3.0 * growth(particle))

print(f"every_run_exits_zero_with_no_error_and_no_warning={all_clean}")
print(f"the_uniform_grid_is_under_resolved_cell_knudsen_below_one="
      f"{under_resolved}")
print(f"and_SPARTA_never_warns_about_it={baseline_never_warns}")
print(f"refining_on_particle_count_does_NOT_improve_the_knudsen_minimum="
      f"{particle_does_not_improve_it}")
print(f"refining_on_the_knudsen_column_does={knudsen_does_improve_it}")
print(f"the_particle_count_criterion_is_self_limiting_not_several_fold="
      f"{particle_criterion_is_self_limiting}")

ok = (all_clean and under_resolved and baseline_never_warns
      and particle_does_not_improve_it and knudsen_does_improve_it
      and particle_criterion_is_self_limiting)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
