"""Tier-2: two numbers a SPARTA deck will quietly get wrong.

  particle_emission:1  the emitted flux is set by the MIXTURE, not by
                       'global nrho'. An 'nrho' keyword on the emit mixture
                       overrides the global value silently, and the domain
                       fills to a steady count set by the mixture's value.
  chemistry:3          Nreact is not the way to tell a cold deck from a broken
                       one. It is a PER-STEP count, so a run with thousands of
                       product particles still reports single digits on every
                       stats line, while a run at a temperature below the
                       activation energy reports the same single digits' worth
                       of zero with a perfectly healthy Ncoll.

Four decks. The emission pair differs by one keyword on one mixture line; the
chemistry pair differs by the mixture temperature and nothing else. The emission
claim is asserted as the RATIO between the two steady counts against the ratio
of the two densities, and the chemistry claim as the presence or absence of
product particles — never as a stored count.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss", "air.species", "air.vss",
                           "air.tce")

EMIT = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o o p
create_box 0 1 0 1 -0.5 0.5
create_grid 20 20 1
global nrho 1e20 fnum 1e15
species ar.species Ar
mixture gas Ar vstream 300 0 0 temp 273.15 {extra}
collide vss gas ar.vss
fix in emit/face gas xlo
timestep 1e-5
stats 400
stats_style step np
run 1200
"""

CHEM = """seed 12345
dimension 2
boundary p p p
global gridcut 0.0
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
global nrho 1e22 fnum 5e11
species air.species N2 O2 N O NO
mixture gas N2 O2 temp {temp}
mixture gas N2 frac 0.8
mixture gas O2 frac 0.2
collide vss species air.vss
react tce air.tce
create_particles gas n 0
compute c count species
timestep 1e-7
stats 200
stats_style step np ncoll nreact c_c[3] c_c[4] c_c[5]
run 1000
"""

ARGON = {"ar.species": DATA["ar.species"], "ar.vss": DATA["ar.vss"]}
AIR = {"air.species": DATA["air.species"], "air.vss": DATA["air.vss"],
       "air.tce": DATA["air.tce"]}


def probe(deck: str, files: dict) -> dict:
    rc, txt = run(deck, files)
    header, rows = stats_rows(txt)

    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "np": column("Np"), "ncoll": column("Ncoll"),
            "nreact": column("Nreact"),
            "products": [sum(v) for v in zip(column("c_c[3]"), column("c_c[4]"),
                                             column("c_c[5]"))]}


global_only = probe(EMIT.format(extra=""), ARGON)
mixture_nrho = probe(EMIT.format(extra="nrho 4e20"), ARGON)
cold = probe(CHEM.format(temp="273.15"), AIR)
hot = probe(CHEM.format(temp="20000.0"), AIR)

for tag, r in (("emit_with_global_nrho_only", global_only),
               ("emit_with_nrho_on_the_mixture", mixture_nrho),
               ("chemistry_below_the_activation_energy", cold),
               ("chemistry_above_it", hot)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")

steady_ratio = ((mixture_nrho["np"][-1] / global_only["np"][-1])
                if global_only["np"] and global_only["np"][-1] else 0.0)
print(f"steady_np_ratio_for_a_4x_mixture_nrho={steady_ratio:.4g}")
print(f"cold_nreact_column={[int(v) for v in cold['nreact']]}")
print(f"hot_nreact_column={[int(v) for v in hot['nreact']]}")
print(f"cold_product_particle_column={[int(v) for v in cold['products']]}")
print(f"hot_product_particle_column={[int(v) for v in hot['products']]}")

both_emit_runs_clean = (global_only["rc"] == 0 and mixture_nrho["rc"] == 0
                        and not global_only["warnings"]
                        and not mixture_nrho["warnings"])
mixture_nrho_wins = 3.0 < steady_ratio < 5.0
both_chem_runs_clean = (cold["rc"] == 0 and hot["rc"] == 0
                        and not cold["warnings"] and not hot["warnings"])
cold_has_healthy_collisions = bool(cold["ncoll"]) and cold["ncoll"][-1] > 100
cold_reacts_never = all(v == 0 for v in cold["nreact"])
cold_makes_no_products = all(v == 0 for v in cold["products"])
hot_makes_products = (bool(hot["products"]) and hot["products"][-1] > 0
                      and hot["products"][-1] > hot["products"][1])
hot_nreact_stays_small = (bool(hot["nreact"])
                          and max(hot["nreact"]) < 0.01 * hot["products"][-1])

print(f"both_emit_runs_exit_zero_with_no_warning={both_emit_runs_clean}")
print(f"an_nrho_on_the_mixture_sets_the_steady_count={mixture_nrho_wins}")
print(f"both_chemistry_runs_exit_zero_with_no_warning={both_chem_runs_clean}")
print(f"the_cold_deck_still_collides_plenty={cold_has_healthy_collisions}")
print(f"the_cold_deck_reports_zero_reactions_and_zero_products="
      f"{cold_reacts_never and cold_makes_no_products}")
print(f"the_hot_deck_accumulates_product_particles={hot_makes_products}")
print(f"but_its_Nreact_column_stays_orders_of_magnitude_smaller="
      f"{hot_nreact_stays_small}")

ok = (both_emit_runs_clean and mixture_nrho_wins and both_chem_runs_clean
      and cold_has_healthy_collisions and cold_reacts_never
      and cold_makes_no_products and hot_makes_products
      and hot_nreact_stays_small)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
