"""Tier-2: a surface heat flux sampled from the first window is not an answer,
and the convergence test is agreement between windows, not the step count.

  hypersonic_flow:2  surface heat flux and pressure need statistical steady
                     state; sampling the first stats block gives a wrong number
                     while the run looks entirely healthy.

The Mach-5 argon cylinder case, tallied with 'compute surf all all etot' through
a 'fix ave/surf' in successive 200-step windows, on two seeds. Note the shapes
this exercises on the way: compute surf is an ARRAY so it is indexed c_s[1],
and a fix ave/surf fed one value is a VECTOR so it is read bare as f_fs.

HOW THE THRESHOLD IS SET. The flux is a Monte-Carlo tally, so the scale is
measured rather than assumed: the noise floor comes from the late windows of ONE
seed, and every assertion is then made against the OTHER seed. Seed B's late
windows must sit inside a small multiple of seed A's floor; both seeds' FIRST
window must sit far outside it.

WHAT THIS CORRECTS. The entry says the first block is "wrong by a large factor".
It is not. Measured, the first window comes out about 1.26 times the converged
plateau — 26 percent, not a factor of several. That still puts it tens of times
outside the seed-to-seed spread, so it is unambiguously wrong and the entry's
advice stands; but an agent told to look for a large factor would see 26 percent,
decide the transient had passed, and sample it anyway. The magnitude is stated as
a percentage now, and the emphasis moved to where it belongs: the discriminator
is that the first window sits outside the seed spread while the late ones agree
inside it.
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

DECK = """seed {seed}
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
compute s surf all all etot
fix fs ave/surf all 1 200 200 c_s[1]
compute r reduce sum f_fs
timestep 1e-4
stats 200
stats_style step np c_r
run 2400
"""


def probe(seed: int) -> dict:
    rc, txt = run(DECK.format(seed=seed), FILES, timeout=1500)
    header, rows = stats_rows(txt)
    flux = (col(header, rows, "c_r")
            if header and rows and "c_r" in header else [])
    # Row 0 is step 0, before the first window has closed; drop it.
    return {"rc": rc, "errors": errors(txt),
            "warnings": [l for l in txt.splitlines() if l.startswith("WARNING")],
            "windows": [v for v in flux[1:] if v != 0.0]}


a = probe(12345)
b = probe(54321)

for tag, r in (("seed_A", a), ("seed_B", b)):
    print(f"{tag}_rc={r['rc']} n_errors={len(r['errors'])} "
          f"n_warnings={len(r['warnings'])}")
    print(f"{tag}_flux_by_window={[round(v, 2) for v in r['windows']]}")

all_clean = (a["rc"] == 0 and b["rc"] == 0 and not a["errors"]
             and not b["errors"] and not a["warnings"] and not b["warnings"])
have = len(a["windows"]) >= 5 and len(b["windows"]) >= 5

LATE = slice(2, None)
a_late = a["windows"][LATE] if have else []
b_late = b["windows"][LATE] if have else []
a_mean = sum(a_late) / len(a_late) if a_late else 0.0
# Noise floor MEASURED on seed A, applied to seed B.
floor = (max(abs(v - a_mean) for v in a_late) / a_mean) if a_late else 0.0
b_dev = [abs(v - a_mean) / a_mean for v in b_late] if b_late else []
a_first_excess = (a["windows"][0] - a_mean) / a_mean if have else 0.0
b_first_excess = (b["windows"][0] - a_mean) / a_mean if have else 0.0

print(f"converged_plateau_from_seed_A_late_windows={a_mean:.6g}")
print(f"measured_noise_floor_from_seed_A={floor:.4g}")
print(f"seed_B_late_window_deviations={[round(v, 5) for v in b_dev]}")
print(f"seed_A_first_window_excess={a_first_excess:.4g}")
print(f"seed_B_first_window_excess={b_first_excess:.4g}")

cross_seed_consistent = bool(b_dev) and max(b_dev) < 3.0 * floor
first_block_far_outside = (floor > 0
                           and a_first_excess > 10.0 * floor
                           and b_first_excess > 10.0 * floor)
# The correction: it is a quarter, not a factor of several.
first_block_is_not_a_large_factor = (0.0 < a_first_excess < 1.0
                                     and 0.0 < b_first_excess < 1.0)

print(f"both_runs_exit_zero_with_no_error_and_no_warning={all_clean}")
print(f"both_runs_produced_enough_windows={have}")
print(f"late_windows_of_the_other_seed_agree_within_the_measured_floor="
      f"{cross_seed_consistent}")
print(f"the_FIRST_window_is_more_than_10x_the_floor={first_block_far_outside}")
print(f"but_it_is_a_fraction_off_NOT_a_large_factor="
      f"{first_block_is_not_a_large_factor}")

ok = (all_clean and have and cross_seed_consistent and first_block_far_outside
      and first_block_is_not_a_large_factor)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
