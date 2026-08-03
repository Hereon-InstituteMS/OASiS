"""Tier-2: `compute temp` is not a thermal temperature in a flow.

The box is fully PERIODIC on purpose: a specularly reflecting wall turns a
uniform drift into two counter-streaming beams, and then compute thermal/grid
correctly reports the beam separation as thermal motion. Uniform drift needs
periodic boundaries.

`compute <ID> temp` sums the kinetic energy of all particles WITHOUT removing
any mean velocity, so it reports T_thermal + m|vstream|^2/(dim*kB). The offset
is proportional to species MASS and to the SQUARE of the stream speed, so it is
huge for a heavy species at high speed and negligible for a slow flow — which
is why "compute temp is always wrong in a flow" would be the wrong lesson.

`compute thermal/grid` removes the per-cell mean and does report a thermal
temperature here. Both runs exit 0 and neither warns.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, run, skip_if_unavailable, stats_rows  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss")

KB = 1.380649e-23
M_AR = 6.63e-26          # kg, from ar.species
T_THERMAL = 273.15

DECK = """seed 12345
dimension 2
boundary p p p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar vstream {v} 0 0 temp {t}
global nrho 7.07043e22 fnum 7.07043e11
collide vss gas ar.vss
create_particles gas n 0
compute tk temp
compute tg thermal/grid all all temp
compute rtg reduce ave c_tg[1]
timestep 1e-9
stats 100
stats_style step np c_tk c_rtg
run 200
"""


def measure(v: float):
    rc, txt = run(DECK.format(v=v, t=T_THERMAL), DATA)
    hdr, rows = stats_rows(txt)
    warn = sum(1 for l in txt.splitlines() if l.upper().startswith("WARNING"))
    return rc, col(hdr, rows, "c_tk")[0], col(hdr, rows, "c_rtg")[-1], warn


rc0, tk0, tg0, w0 = measure(0.0)
rc1, tk1, tg1, w1 = measure(1000.0)
rc2, tk2, tg2, w2 = measure(10.0)

pred = M_AR * 1000.0 ** 2 / (3.0 * KB)      # SPARTA divides by dim=3 always

print(f"vstream_0_compute_temp={tk0:.3f}")
print(f"vstream_0_thermal_grid={tg0:.3f}")
print(f"vstream_1000_compute_temp={tk1:.3f}")
print(f"vstream_1000_thermal_grid={tg1:.3f}")
print(f"vstream_10_compute_temp={tk2:.3f}")
print(f"predicted_offset_m_v2_over_3kB={pred:.1f}")
print(f"measured_offset={tk1 - tk0:.1f}")
print(f"all_runs_exit_zero={rc0 == 0 and rc1 == 0 and rc2 == 0}")
print(f"no_warnings={w0 == 0 and w1 == 0 and w2 == 0}")
print(f"offset_matches_m_v2_over_3kB={abs((tk1 - tk0) - pred) / pred < 0.05}")
print(f"thermal_grid_stays_near_thermal={abs(tg1 - tg0) / max(tg0, 1) < 0.15}")
print(f"slow_flow_offset_is_negligible={abs(tk2 - tk0) < 1.0}")

ok = (rc0 == 0 and rc1 == 0 and rc2 == 0 and w0 == w1 == w2 == 0
      and abs((tk1 - tk0) - pred) / pred < 0.05
      and abs(tg1 - tg0) / max(tg0, 1) < 0.15
      and abs(tk2 - tk0) < 1.0)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
