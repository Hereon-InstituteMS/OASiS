"""Tier-2: forgetting `collide` in a SPARTA deck is silent, and wrong by 8x.

This is the flagship "clean run, meaningless physics" DSMC pitfall. The wrong
variant differs from the right one by ONE deleted line; both exit 0, both print
a healthy stats table, and neither warns.

Case: 2d argon Fourier channel. boundary p ss p, box 2e-5 x 1e-4 m, 4x60 grid,
nrho 7.07043e23, fnum 1e11, ylo wall diffuse 300 K, yhi wall diffuse 1000 K,
dt 1e-9 s, 3000 steps. Wall energy flux via compute boundary + fix ave/time
(mode vector, which compute boundary requires).

Measured 2026-08-03 with SPARTA (24 Sep 2025), spa_serial:
    with    collide vss : cold wall ~2.0e5  W/m2, mid-gas T ~657 K
    without collide     : cold wall ~1.78e6 W/m2, mid-gas T ~535 K   (~8.4x)
and Ncoll is identically 0 in the collisionless run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = [
    os.environ.get("SPARTA_BINARY"),
    shutil.which("spa_serial"),
    shutil.which("spa_mpi"),
    "/home/alexander/Schreibtisch/sparta/src/spa_serial",
    str(Path.home() / "sparta" / "src" / "spa_serial"),
]
BINARY = next((c for c in CANDIDATES if c and Path(c).is_file()), None)
if BINARY is None:
    print("SKIP: no SPARTA binary found (set SPARTA_BINARY)")
    sys.exit(0)

DECK = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary p ss p
create_box 0 2.0e-5 0 1.0e-4 -0.5 0.5
create_grid 4 60 1
species ar.species Ar
mixture air Ar vstream 0 0 0 temp 650.0
global nrho 7.07043e23 fnum 1.0e11
surf_collide cold diffuse 300.0 1.0
surf_collide hot diffuse 1000.0 1.0
bound_modify ylo collide cold
bound_modify yhi collide hot
{collide}create_particles air n 0
compute b boundary all ke etot n
fix bav ave/time 1 250 250 c_b[*] mode vector
compute ct thermal/grid all all temp
compute tmid reduce ave c_ct[1]
stats 250
stats_style step np ncoll f_bav[3][1] f_bav[4][1] c_tmid
timestep 1.0e-9
run 3000
"""


def run(with_collide: bool):
    work = Path(tempfile.mkdtemp(prefix="sparta_collide_"))
    for f in ("ar.species", "ar.vss"):
        shutil.copy(HERE / f, work / f)
    (work / "in.case").write_text(
        DECK.format(collide="collide vss air ar.vss\n" if with_collide else ""))
    proc = subprocess.run([BINARY, "-in", "in.case"], cwd=str(work),
                          capture_output=True, text=True, timeout=600)
    log = (work / "log.sparta").read_text() if (work / "log.sparta").is_file() else ""
    rows = []
    started = False
    for line in log.splitlines():
        if line.startswith("Step "):
            started = True
            continue
        if started:
            parts = line.split()
            if len(parts) == 6:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    break
            else:
                break
    shutil.rmtree(work, ignore_errors=True)
    return proc.returncode, rows, log.splitlines()[0] if log else ""


rc_on, rows_on, ver = run(True)
rc_off, rows_off, _ = run(False)

if not rows_on or not rows_off:
    print("FAIL: no stats table parsed")
    sys.exit(1)

# average the last four stats rows (steady state)
def steady(rows, col):
    tail = rows[-4:]
    return sum(r[col] for r in tail) / len(tail)


flux_on, flux_off = steady(rows_on, 3), steady(rows_off, 3)
t_on, t_off = steady(rows_on, 5), steady(rows_off, 5)
ncoll_on = max(r[2] for r in rows_on)
ncoll_off = max(r[2] for r in rows_off)

print(f"sparta_version_line={ver.strip()}")
print(f"rc_with_collide={rc_on}")
print(f"rc_without_collide={rc_off}")
print(f"both_runs_exit_zero={rc_on == 0 and rc_off == 0}")
print(f"cold_wall_flux_with_collide={flux_on:.4e}")
print(f"cold_wall_flux_without_collide={flux_off:.4e}")
print(f"flux_ratio_without_over_with={flux_off / flux_on:.3f}")
print(f"mid_gas_T_with_collide={t_on:.1f}")
print(f"mid_gas_T_without_collide={t_off:.1f}")
print(f"max_ncoll_with_collide={ncoll_on:.0f}")
print(f"max_ncoll_without_collide={ncoll_off:.0f}")
print(f"ncoll_identically_zero_without_collide={ncoll_off == 0}")
print(f"flux_overestimated_by_more_than_5x={flux_off / flux_on > 5.0}")

ok = (rc_on == 0 and rc_off == 0 and ncoll_on > 0 and ncoll_off == 0
      and flux_off / flux_on > 5.0 and abs(t_on - t_off) > 50.0)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
