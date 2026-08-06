"""Tier-2: the thermostat ramps across the RUN, and the histogram silently
drops everything outside its range.

  collision_relaxation:6  'fix temp/rescale Nevery Tstart Tstop' interpolates
                          its target linearly from the run's first step to its
                          last, so the gas follows a straight line rather than
                          relaxing; 'fix temp/global/rescale' takes a fourth
                          argument and a different signature.
  collision_relaxation:7  'fix ave/histo' refuses per-particle input in the
                          default scalar mode; in scalar mode on a GLOBAL
                          compute it holds Nrepeat samples, not the particle
                          population; and element [2] of its global 4-vector is
                          the count that fell OUTSIDE [lo,hi] and was dropped
                          from every bin without a warning.

The ramp is asserted as a STRAIGHT LINE and as landing on the Tstop the deck
set — a shape and a deck input, not a stored temperature — and both are judged
against a NOISE FLOOR measured from three other seeds, with the ramp's own rise
required to be far above that floor before anything else is read. The histogram
claims are counts compared against each other: in-range plus out-of-range equals
Nrepeat times the particle count exactly, and narrowing the range moves counts
from one column to the other without changing the total.

Mutation control: T2_MUTATE=1 widens the histogram range so that nothing falls
outside it, leaving the sample schedule, the box, the seed and the run length
identical. The dropped-count column goes to zero and
`values_outside_the_range_are_dropped_silently` and
`narrowing_the_range_moves_counts_into_the_dropped_column` go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

BOX = """seed {seed}
dimension 2
boundary rr rr p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e11
create_particles gas n 0
collide vss gas ar.vss
timestep 1e-8
compute gt temp
{body}
stats {stats}
stats_style step np {cols}
run {nsteps}
"""


def go(body: str, cols: str, stats: int = 100, nsteps: int = 500,
       seed: int = 12345):
    rc, txt = run(BOX.format(seed=seed, body=body, cols=cols, stats=stats,
                             nsteps=nsteps), DATA)
    h, r = stats_rows(txt)
    return rc, errors(txt), (h, r)


if MUTATE:
    print("mutation=histogram_range_widened_so_nothing_falls_outside_it")

# collision_relaxation:6 — the linear ramp. SPARTA is Monte Carlo, so the
# straightness of the line is judged against a NOISE FLOOR, and the ramp is run
# on a seed that is not among the ones the floor was measured from. Everything
# below is a RATIO to that floor; no temperature is pinned.
#
# WHERE THE FLOOR COMES FROM, because the obvious choice is wrong here: in a
# reflecting box with no thermostat, VSS collisions conserve total kinetic
# energy exactly, so 'compute temp' is CONSTANT across the run to every digit
# the table prints and the within-run spread is identically zero. The
# fluctuation that matters is between REALISATIONS — the draw that sets the
# initial velocities — so the floor is the spread of that constant across three
# seeds.
TSTART, TSTOP = 300.0, 600.0
FLOOR_SEEDS = (99131, 99137, 99149)

floor_vals, floor_ok = [], True
for s in FLOOR_SEEDS:
    rc_f, e_f, (h_f, r_f) = go("", "c_gt", seed=s)
    if rc_f or e_f or not r_f:
        floor_ok = False
        break
    floor_vals.append(col(h_f, r_f, "c_gt")[-1])
floor = (max(floor_vals) - min(floor_vals)) if len(floor_vals) == 3 else 0.0
print(f"the_noise_floor_deck_runs={floor_ok and floor > 0.0}")

rc_t, e_t, (h_t, r_t) = go(f"fix tr temp/rescale 20 {TSTART} {TSTOP}", "c_gt")
temps = col(h_t, r_t, "c_gt") if (h_t and r_t) else []
print(f"the_thermostat_deck_runs={rc_t == 0 and not e_t and len(temps) > 3}")

# The rise is the SIGNAL and must dwarf the floor, or nothing below means
# anything.
rise = (temps[-1] - temps[0]) if len(temps) > 3 else 0.0
print(f"the_rise_is_far_above_the_noise_floor="
      f"{floor > 0.0 and rise > 20 * floor}")

# Straightness: residuals about the line through the first and last row, judged
# against the same floor.
straight = False
if len(temps) > 3 and floor > 0.0:
    n = len(temps) - 1
    line = [temps[0] + (temps[-1] - temps[0]) * i / n for i in range(n + 1)]
    straight = max(abs(a - b) for a, b in zip(temps, line)) < 2 * floor
print(f"the_target_is_ramped_linearly_across_the_run={straight}")

# and it ends at the Tstop the DECK set — again to within the floor, not to a
# stored value.
lands = bool(temps) and floor > 0.0 and abs(temps[-1] - TSTOP) < 3 * floor
print(f"it_lands_on_tstop_at_the_last_step={lands}")

rc_g, e_g, _ = go(f"fix tr temp/global/rescale 20 1 1 1 {TSTART} {TSTOP}", "")
GLOBAL_MSG = ("ERROR: Illegal fix temp/global/rescale command "
              "(../fix_temp_global_rescale.cpp:29)")
print(f"temp_global_rescale_has_a_different_signature="
      f"{any(GLOBAL_MSG in e for e in e_g)}")
rc_g2, e_g2, _ = go(f"fix tr temp/global/rescale 20 {TSTART} {TSTOP} 0.5", "")
print(f"its_fourth_argument_is_a_fraction={rc_g2 == 0 and not e_g2}")
rc_r2, e_r2, _ = go(f"fix tr temp/rescale 20 {TSTOP}", "")
RESCALE_MSG = ("ERROR: Illegal fix temp/rescale command "
               "(../fix_temp_rescale.cpp:31)")
print(f"temp_rescale_needs_three_numbers={any(RESCALE_MSG in e for e in e_r2)}")

# collision_relaxation:7 — the histogram.
NREPEAT = 5
rc_p, e_p, _ = go("compute kp ke/particle\n"
                  f"fix h ave/histo 10 {NREPEAT} 100 0 2e-20 20 c_kp", "")
PERPART_MSG = ("ERROR: Fix ave/histo cannot input per-particle values in "
               "scalar mode (../fix_ave_histo.cpp:228)")
print(f"per_particle_input_needs_mode_vector="
      f"{any(PERPART_MSG in e for e in e_p)}")

rc_a, e_a, _ = go(f"fix h ave/histo 10 {NREPEAT} 100 c_gt", "")
ARG_MSG = "ERROR: Illegal fix ave/histo command (../fix_ave_histo.cpp:53)"
print(f"lo_hi_and_nbin_are_required={any(ARG_MSG in e for e in e_a)}")

# In scalar mode on a GLOBAL compute the histogram holds Nrepeat samples.
rc_s, e_s, (h_s, r_s) = go(
    f"fix h ave/histo 10 {NREPEAT} 100 200 400 20 c_gt",
    "f_h[1] f_h[2]")
n_in = col(h_s, r_s, "f_h[1]") if (h_s and r_s) else []
print(f"a_global_compute_gives_nrepeat_samples="
      f"{len(n_in) > 1 and all(v == float(NREPEAT) for v in n_in[1:])}")


def histo(lo: float, hi: float):
    rc, e, (h, r) = go("compute kp ke/particle\n"
                       f"fix h ave/histo 10 {NREPEAT} 100 {lo} {hi} 20 "
                       "c_kp mode vector", "f_h[1] f_h[2]")
    if rc or not r:
        return None
    return col(h, r, "f_h[1]"), col(h, r, "f_h[2]"), col(h, r, "Np")


# A range that clips the tail, and (under mutation) one that does not.
NARROW_HI = 1e-18 if MUTATE else 2e-20
wide = histo(0.0, 1e-18)
narrow = histo(0.0, NARROW_HI)
print(f"the_histogram_decks_run={wide is not None and narrow is not None}")

conserved = dropped = moved = False
if wide and narrow:
    win, wout, wnp = wide
    nin, nout, nnp = narrow
    # in + out is the whole sample: Nrepeat windows times the particle count.
    conserved = all(a + b == NREPEAT * n
                    for a, b, n in zip(win[1:], wout[1:], wnp[1:]))
    dropped = any(v > 0 for v in nout[1:])
    moved = all(a >= b for a, b in zip(nout[1:], wout[1:])) and dropped
print(f"in_range_plus_out_of_range_is_the_whole_sample={conserved}")
print(f"values_outside_the_range_are_dropped_silently={dropped}")
print(f"narrowing_the_range_moves_counts_into_the_dropped_column={moved}")

ok = (floor_ok and floor > 0.0 and rc_t == 0 and not e_t
      and rise > 20 * floor and straight and lands
      and any(GLOBAL_MSG in e for e in e_g) and rc_g2 == 0 and not e_g2
      and any(RESCALE_MSG in e for e in e_r2)
      and any(PERPART_MSG in e for e in e_p) and any(ARG_MSG in e for e in e_a)
      and len(n_in) > 1 and all(v == float(NREPEAT) for v in n_in[1:])
      and conserved and dropped and moved)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
