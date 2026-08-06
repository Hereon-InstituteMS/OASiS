"""Tier-2: two ways a SPARTA wall model is wrong without SPARTA objecting.

  surface_interaction:0  'surf_collide <ID> diffuse' takes Tsurf then
                         accommodation. A SWAP is caught only when the
                         accommodation ends up outside [0,1]: 'diffuse 0.5 300'
                         aborts, but 'diffuse 0.9 0.3' is a legal 0.9 KELVIN
                         wall and runs with no diagnostic whatever.
  surface_interaction:1  'surf_collide <ID> specular' reverses the normal
                         velocity and so transfers NO energy: the per-surf etot
                         collapses to round-off while the collision count stays
                         healthy. Passing it a temperature is an error.

The specular assertion is deliberately written as a MAGNITUDE RATIO against the
diffuse control run in the same script, not against a stored number. A DSMC wall
tally is a Monte-Carlo sum: its value moves with the seed, and its sign moves
too. What does not move is that round-off is many orders of magnitude below a
physical flux, so that is what is asserted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss", "data.circle")

DECK = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o o p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
global nrho 1e20 fnum 1e17
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15
read_surf data.circle
surf_collide cw {args}
surf_modify all collide cw
create_particles gas n 0
collide vss gas ar.vss
compute e surf all all etot
fix ae ave/surf all 10 10 100 c_e[1]
compute re reduce sum f_ae
timestep 1e-4
stats 100
stats_style step np nscoll c_re
run 400
"""
# The timestep is deliberately coarse for the box. This fixture tests wall-model
# BOOKKEEPING — which argument means what, and whether energy is transferred at
# all — not transport accuracy, and a small step simply means too few particles
# reach the circle for the etot column to be worth reading.


def probe(args: str) -> dict:
    rc, txt = run(DECK.format(args=args), DATA)
    header, rows = stats_rows(txt)
    def column(name):
        return (col(header, rows, name)
                if (header and rows and name in header) else [])
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "etot": column("c_re"), "nscoll": column("Nscoll")}


CASES = {
    "diffuse_correct_300_and_1.0": "diffuse 300 1.0",
    "diffuse_swapped_accom_out_of_range": "diffuse 0.5 300",
    "diffuse_swapped_both_in_range": "diffuse 0.9 0.3",
    "diffuse_tsurf_zero": "diffuse 0.0 1.0",
    "diffuse_accommodation_negative": "diffuse 300 -0.1",
    "specular_no_argument": "specular",
    "specular_given_a_temperature": "specular 300",
}
r = {name: probe(args) for name, args in CASES.items()}

for name, res in r.items():
    peak = max((abs(v) for v in res["etot"]), default=0.0)
    print(f"{name}_rc={res['rc']} n_errors={len(res['errors'])} "
          f"n_warnings={len(res['warnings'])} peak_abs_etot={peak:.4g} "
          f"max_nscoll={max(res['nscoll'], default=0):.0f}")
    if res["errors"]:
        print(f"{name}_message={res['errors'][0]}")

QUOTED = {
    "diffuse_swapped_accom_out_of_range":
        "ERROR: Illegal surf_collide diffuse command "
        "(../surf_collide_diffuse.cpp:50)",
    "diffuse_accommodation_negative":
        "ERROR: Illegal surf_collide diffuse command "
        "(../surf_collide_diffuse.cpp:50)",
    "diffuse_tsurf_zero":
        "ERROR: Surf_collide tsurf <= 0.0 (../surf_collide.cpp:125)",
    "specular_given_a_temperature":
        "ERROR: Illegal surf_collide specular command "
        "(../surf_collide_specular.cpp:42)",
}
quoted_ok = all(QUOTED[k] in r[k]["errors"] for k in QUOTED)
for k in QUOTED:
    if QUOTED[k] not in r[k]["errors"]:
        print(f"UNEXPECTED: {k} printed {r[k]['errors'][:1]} "
              f"not {QUOTED[k]!r}")

silent_swap = r["diffuse_swapped_both_in_range"]
diffuse = r["diffuse_correct_300_and_1.0"]
specular = r["specular_no_argument"]

peak_diffuse = max((abs(v) for v in diffuse["etot"]), default=0.0)
peak_specular = max((abs(v) for v in specular["etot"]), default=0.0)
ratio = (peak_diffuse / peak_specular) if peak_specular else float("inf")

print(f"diffuse_over_specular_peak_etot_ratio={ratio:.4g}")
print(f"every_quoted_surf_collide_message_is_verbatim={quoted_ok}")
print(f"in_range_argument_swap_is_completely_silent="
      f"{silent_swap['rc'] == 0 and not silent_swap['errors'] and not silent_swap['warnings']}")
print(f"specular_wall_energy_is_round_off_vs_diffuse={ratio > 1e6}")
print(f"specular_wall_still_registers_collisions="
      f"{max(specular['nscoll'], default=0) > 0}")
print(f"diffuse_control_transfers_real_energy={peak_diffuse > 1.0}")

ok = (quoted_ok
      and silent_swap["rc"] == 0 and not silent_swap["errors"]
      and not silent_swap["warnings"]
      and ratio > 1e6
      and max(specular["nscoll"], default=0) > 0
      and peak_diffuse > 1.0)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
