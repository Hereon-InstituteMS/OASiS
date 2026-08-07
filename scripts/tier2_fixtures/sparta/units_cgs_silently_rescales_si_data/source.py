"""Tier-2: 'units cgs' on a deck whose data files are SI is silent.

The two decks differ by ONE leading line. SPARTA's units command changes only
the value of Boltzmann's constant it multiplies your numbers by; it does not
convert the shipped ar.species / ar.vss data, which are SI. Both runs exit 0
and print no warning, the particle count and the compute-temp column are
identical, and only the collision statistics and the Cell-touches diagnostic
betray that the physics has changed.

Mutation control: T2_MUTATE=1 removes the `units cgs` line from the run that is
supposed to carry it, so both decks are SI and the two runs are byte-identical.
`cgs_collision_rate_is_far_higher` and `cell_touches_tell_present` both go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, run, skip_if_unavailable, stats_rows,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

DECK = """{units}seed 12345
dimension 2
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15
global nrho 7.07043e22 fnum 7.07043e11
collide vss gas ar.vss
create_particles gas n 0
compute tk temp
timestep 1e-9
stats 100
stats_style step np nattempt ncoll c_tk
run 200
"""


def measure(units_line: str):
    rc, txt = run(DECK.format(units=units_line), DATA)
    hdr, rows = stats_rows(txt)
    ncoll = max(col(hdr, rows, "Ncoll")) if rows else -1
    temp = col(hdr, rows, "c_tk")[0] if rows else -1
    touches = 0.0
    for line in txt.splitlines():
        if line.startswith("Cell-touches/particle/step:"):
            touches = float(line.split(":")[1])
    warn = sum(1 for l in txt.splitlines() if l.upper().startswith("WARNING"))
    return rc, ncoll, temp, touches, warn, txt


rc_si, nc_si, t_si, tc_si, w_si, _ = measure("")
# Under mutation the unit-system mismatch is REMOVED: the second deck is SI too,
# so the collision rate and Cell-touches tells have nothing to separate.
rc_cgs, nc_cgs, t_cgs, tc_cgs, w_cgs, _ = measure("" if MUTATE else "units cgs\n")
if MUTATE:
    print("mutation=units_cgs_line_removed_so_both_decks_are_SI")

print(f"rc_si={rc_si}")
print(f"rc_cgs={rc_cgs}")
print(f"both_runs_exit_zero={rc_si == 0 and rc_cgs == 0}")
print(f"max_ncoll_si={nc_si:.0f}")
print(f"max_ncoll_cgs={nc_cgs:.0f}")
print(f"cgs_collision_rate_is_far_higher={nc_cgs > 100 * max(nc_si, 1)}")
print(f"compute_temp_si={t_si:.5f}")
print(f"compute_temp_cgs={t_cgs:.5f}")
print(f"compute_temp_is_identical={abs(t_si - t_cgs) < 1e-6}")
print(f"cell_touches_si={tc_si:.4f}")
print(f"cell_touches_cgs={tc_cgs:.4f}")
print(f"cell_touches_tell_present={tc_cgs > 100 * tc_si}")
print(f"warnings_si={w_si} warnings_cgs={w_cgs}")
print(f"no_warning_in_either_log={w_si == 0 and w_cgs == 0}")

ok = (rc_si == 0 and rc_cgs == 0 and nc_cgs > 100 * max(nc_si, 1)
      and abs(t_si - t_cgs) < 1e-6 and tc_cgs > 100 * tc_si
      and w_si == 0 and w_cgs == 0)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
