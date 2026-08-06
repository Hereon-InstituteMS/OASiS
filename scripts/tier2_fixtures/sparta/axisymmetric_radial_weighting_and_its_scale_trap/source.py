"""Tier-2: radial weighting evens out axisymmetric cell occupancy — and the
same command silently empties the domain on a small one.

  axisymmetric:4  in an axisymmetric run the cell volume grows linearly with
                  radius, so with uniform weighting the near-axis cells hold
                  very few particles while the outer cells are over-resolved;
                  'global weight cell radius' is what that is for.

WHAT THIS ADDS. The entry treats 'global weight cell radius' as a free remedy.
It is not, and the reason is in grid.cpp:2032: for an axisymmetric run the
weight is set to the cell's MEAN RADIUS IN RUN LENGTH UNITS —
0.5*(hi[1]+lo[1]) — not to a ratio against a reference radius. create_particles
then divides each cell's volume by that weight, so on a domain whose radius is
small in metres every weight is far below 1 and the requested particle count is
inflated by roughly 1/r.

Executed on a millimetre box, create_particles does not complete at all. On a
metre box with a small fnum it does something worse than failing: it prints
'WARNING: Created unexpected # of particles: 0 versus <N>' and then 'Created 0
particles', exits 0, and runs to completion with an EMPTY domain. Nothing
aborts. So the honest signal for the scale trap is a warning most drivers do not
read and a stats table of zeros.

The occupancy comparison itself is asserted as a RATIO of the per-cell maximum
to the per-cell minimum, which is dimensionless and independent of fnum. The two
decks deliberately use different fnum values — that is the only way to get
comparable total counts once the weighting has changed the effective volume —
and the assertion is chosen so that difference cannot affect it.

Mutation control: T2_MUTATE=1 adds `weight cell radius` to the UNIFORM deck and
changes nothing else — same box, same nrho, same fnum, same grid, same seed. The
graded occupancy that `global weight cell radius` exists to remove is then gone
from the deck that is supposed to show it, and the max/min spread lands on the
value the radially weighted deck measures independently (about 1.0, against 49.5
unmutated), not merely somewhere else.
`uniform_weighting_grades_occupancy_by_more_than_10x` and
`radial_weighting_removes_the_gradient` go False. The scale-trap probe is left
alone: its pathology is a request that cannot be met, not the weighting.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

DECK = """seed 12345
dimension 2
boundary p ar p
global gridcut 0.0
create_box 0 {L} 0 {L} -0.5 0.5
create_grid 20 20 1
global nrho 1e16 fnum {fnum} {weight}
species ar.species Ar
mixture gas Ar temp 273.15
create_particles gas n 0
compute npc grid all all n
compute nmin reduce min c_npc[1]
compute nmax reduce max c_npc[1]
compute nave reduce ave c_npc[1]
stats 1
stats_style step np c_nmin c_nmax c_nave
run 0
"""

WEIGHT = "weight cell radius"


def probe(L: str, fnum: str, weight: str, timeout: int = 120) -> dict:
    rc, txt = run(DECK.format(L=L, fnum=fnum, weight=weight), DATA,
                  timeout=timeout)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    created = re.search(r"Created (\d+) particles", txt)
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l.strip() for l in txt.splitlines()
                         if l.startswith("WARNING")],
            "created": int(created.group(1)) if created else -1,
            "nmin": column("c_nmin"), "nmax": column("c_nmax"),
            "np": column("Np")}


# --------------------------------------------------------- axisymmetric:4
# Metre-scale box, where the radial weight is order 1 and both decks complete.
# Under mutation the pathology is REMOVED: the "uniform" deck is radially
# weighted too, at its own unchanged fnum.
if MUTATE:
    print("mutation=uniform_deck_given_global_weight_cell_radius")
uniform = probe("1.0", "1.57e12", WEIGHT if MUTATE else "")
radial = probe("1.0", "6.3e13", WEIGHT)

# --------------------------------------------- the scale trap, same command
# Same command on a metre box with a small fnum: the inflated request cannot be
# met, and SPARTA warns instead of failing.
starved = probe("1.0", "1e6", WEIGHT)

for tag, r in (("uniform_weighting", uniform),
               ("radial_weighting", radial),
               ("radial_weighting_with_an_unmet_request", starved)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])} created={r['created']}")
    if r["warnings"]:
        print(f"{tag}_warning={r['warnings'][0]}")

print(f"uniform_cell_min={uniform['nmin'][-1]:.4g} "
      f"cell_max={uniform['nmax'][-1]:.4g}")
print(f"radial_cell_min={radial['nmin'][-1]:.4g} "
      f"cell_max={radial['nmax'][-1]:.4g}")


def spread(r):
    lo = r["nmin"][-1] if r["nmin"] else 0.0
    hi = r["nmax"][-1] if r["nmax"] else 0.0
    return (hi / lo) if lo else float("inf")


uniform_spread = spread(uniform)
radial_spread = spread(radial)
print(f"uniform_max_over_min={uniform_spread:.4g}")
print(f"radial_max_over_min={radial_spread:.4g}")

both_complete = (uniform["rc"] == 0 and radial["rc"] == 0
                 and uniform["created"] > 0 and radial["created"] > 0)
# The mechanism, not a number: occupancy tracks radius, so on an N-cell grid the
# outer/inner ratio is of order 2N-1. Asserted as "more than an order of
# magnitude" so it does not encode this grid's cell count.
uniform_is_strongly_graded = uniform_spread > 10.0
radial_is_nearly_flat = radial_spread < 1.5
weighting_removes_the_gradient = radial_spread < 0.1 * uniform_spread

starved_warned = any("Created unexpected # of particles" in w
                     for w in starved["warnings"])
starved_is_silent_otherwise = (starved["rc"] == 0 and not starved["errors"])
starved_domain_is_empty = (starved["created"] == 0
                           and bool(starved["np"]) and starved["np"][-1] == 0)

print(f"both_weightings_complete_on_a_metre_box={both_complete}")
print(f"uniform_weighting_grades_occupancy_by_more_than_10x="
      f"{uniform_is_strongly_graded}")
print(f"radial_weighting_makes_occupancy_nearly_uniform={radial_is_nearly_flat}")
print(f"radial_weighting_removes_the_gradient={weighting_removes_the_gradient}")
print(f"an_unmet_radial_request_only_WARNS={starved_warned}")
print(f"and_otherwise_exits_zero_with_no_error={starved_is_silent_otherwise}")
print(f"leaving_the_domain_completely_empty={starved_domain_is_empty}")

ok = (both_complete and uniform_is_strongly_graded and radial_is_nearly_flat
      and weighting_removes_the_gradient and starved_warned
      and starved_is_silent_otherwise and starved_domain_is_empty)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
