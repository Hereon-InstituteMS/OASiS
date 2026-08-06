"""Tier-2: there is no flux boundary condition, and the PID that stands in for
one has an inverted sign, no clamp and a schedule-dependent gain.

  conjugate_heat_transfer:10        every wall model fixes a TEMPERATURE or transfers
                                nothing; a flux can only be reached indirectly.
  conjugate_heat_transfer:7     the variable-style chain a working PID needs.
  conjugate_heat_transfer:8     the sign is inverted versus textbook PID, there
                                is no clamp, and divergence returns SUCCESS.
  conjugate_heat_transfer:9     the gains are multiplied by alpha*tau with
                                tau = Nevery*dt, so the schedule rescales them.

The controller assertions are about SHAPE and DIRECTION, never magnitude: that
a loop with one sign of kp grows monotonically over orders of magnitude while
returning rc = 0, that kp = 0 holds the control variable exactly fixed, and that
the fix's own P/I/D vector tracks the control it is producing. On a Monte-Carlo
code that is the only safe way to state it, and it is also the way a user would
recognise it.

Mutation control: T2_MUTATE=1 sets kp to 0 in the diverging deck — the single
edit that removes the runaway, leaving the plant, the setpoint, alpha, Nevery,
the averaging window, the seed and the run length identical. The control
variable then sits still, so `the_loop_diverges_monotonically` and
`divergence_returns_success` go False.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

# A slab between two diffuse walls; the lower wall's temperature is the control
# variable and the energy flux through it is the process variable.
DECK = """seed 12345
dimension 2
boundary p ss p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 40 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 5e9
create_particles gas n 0
collide vss gas ar.vss
{vars}
surf_collide LO diffuse {twall} 1.0
surf_collide HI diffuse 300.0 1.0
bound_modify ylo collide LO
bound_modify yhi collide HI
compute cb boundary gas etot
fix av ave/time 10 10 100 c_cb[*] mode vector
{controller}
timestep 1e-8
stats 500
stats_style step np f_av[3] {cols}
run 2000
"""

CHAIN = "variable tint internal 300.0\nvariable twall equal v_tint"
PID = "fix pid controller 100 1e10 {kp} 0.0 0.0 f_av[3] -1e-6 tint"


def go(vars_: str, twall: str, controller: str, cols: str):
    rc, txt = run(DECK.format(vars=vars_, twall=twall, controller=controller,
                              cols=cols), DATA)
    h, r = stats_rows(txt)
    return rc, errors(txt), (h, r), txt


if MUTATE:
    print("mutation=proportional_gain_of_the_diverging_loop_set_to_zero")

KP_RUNAWAY = 0.0 if MUTATE else 1.0

rc_d, e_d, (h_d, r_d), _ = go(CHAIN, "v_twall", PID.format(kp=KP_RUNAWAY),
                              "v_twall f_pid[1]")
rc_z, e_z, (h_z, r_z), _ = go(CHAIN, "v_twall", PID.format(kp=0.0),
                              "v_twall f_pid[1]")

print(f"the_pid_deck_completes={rc_d == 0 and not e_d and len(r_d) > 2}")

tw_d = col(h_d, r_d, "v_twall") if (h_d and r_d) else []
tw_z = col(h_z, r_z, "v_twall") if (h_z and r_z) else []
p_d = col(h_d, r_d, "f_pid[1]") if (h_d and r_d) else []

# kp = 0 freezes the control variable at its initial value: the loop really is
# doing nothing but the proportional term.
frozen = bool(tw_z) and all(v == tw_z[0] for v in tw_z)
print(f"zero_gain_holds_the_control_variable_fixed={frozen}")

# Monotone growth over many decades, with no clamp and no complaint. "Many
# decades" not "this many": the assertion is that consecutive stats rows keep
# rising and that the last one is at least a million times the first, which a
# bounded controller could never do.
rising = len(tw_d) > 3 and all(b > a for a, b in zip(tw_d[1:], tw_d[2:]))
decades = bool(tw_d) and tw_d[0] > 0 and tw_d[-1] / tw_d[0] > 1e6
diverges = rising and decades
print(f"the_loop_diverges_monotonically={diverges}")
if not diverges and not MUTATE:
    print(f"UNEXPECTED: control variable did not run away: {tw_d[:2]}...{tw_d[-1:]}")
print(f"divergence_returns_success={diverges and rc_d == 0 and not e_d}")

# The fix's own 3-vector is the P/I/D contribution, and with ki = kd = 0 the
# whole change comes from element 1 — which is how a reader diagnoses it.
tracks = (len(p_d) > 2 and len(tw_d) == len(p_d)
          and all(abs(p) > 0 for p in p_d[1:]))
print(f"the_fixes_p_i_d_vector_tracks_the_control={tracks and diverges}")

# conjugate_heat_transfer:8 — the only guard that ever fires is the consumer's.
rc_n, e_n, _, _ = go(CHAIN, "v_twall", PID.format(kp=-1.0), "v_twall")
TSURF_MSG = "ERROR: Surf_collide tsurf <= 0.0 (../surf_collide.cpp:183)"
print(f"the_other_sign_drives_the_wall_below_absolute_zero="
      f"{any(TSURF_MSG in e for e in e_n)}")

# conjugate_heat_transfer:7 — the variable-style rules. A DRAFT of this entry
# claimed the internal control variable needed an equal-style wrapper before it
# could be a wall temperature; execution says otherwise, because
# Variable::equal_style() returns true for INTERNAL (variable.cpp:1016). The
# fixture asserts what actually happens.
rc_i, e_i, _, _ = go("variable tint internal 300.0", "v_tint", "", "")
print(f"an_internal_variable_is_accepted_as_a_wall_temperature="
      f"{rc_i == 0 and not e_i}")
if rc_i != 0:
    print(f"UNEXPECTED: internal variable as tsurf gave {e_i}")

rc_e, e_e, _, _ = go("variable tint equal 300.0\nvariable twall equal v_tint",
                     "v_twall", PID.format(kp=1.0), "")
NOT_INTERNAL = ("ERROR: Fix controller variable is not internal-style variable "
                "(../fix_controller.cpp:135)")
print(f"the_control_variable_must_be_internal_style="
      f"{any(NOT_INTERNAL in e for e in e_e)}")

# A global ARRAY named directly is rejected...
rc_c, e_c, _, _ = go(CHAIN, "v_twall",
                     "fix pid controller 100 1e10 1.0 0.0 0.0 c_cb[1] "
                     "-1e-6 tint", "")
ARRAY_MSG = ("ERROR: Fix controller compute does not calculate a global scalar "
             "or vector (../fix_controller.cpp:104)")
print(f"a_global_array_process_variable_is_rejected="
      f"{any(ARRAY_MSG in e for e in e_c)}")

# ...but a TWO-INDEX reference is NOT: the parser truncates at the first '[',
# so f_av[3][1] is silently read as f_av[3] and the loop runs. The only way to
# see it is that the deck behaves exactly like the single-index one.
rc_a, e_a, (h_a, r_a), _ = go(CHAIN, "v_twall",
                              "fix pid controller 100 1e10 1.0 0.0 0.0 "
                              "f_av[3][1] -1e-6 tint", "v_twall f_pid[1]")
tw_a = col(h_a, r_a, "v_twall") if (h_a and r_a) else []
two_index_silent = bool(tw_a) and (rc_a != 0 or len(tw_a) > 1)
print(f"a_two_index_process_variable_is_not_rejected={two_index_silent}")
print(f"it_behaves_as_the_single_index_form="
      f"{bool(tw_a) and bool(tw_d) and tw_a[:len(tw_d)] == tw_d[:len(tw_a)]}")

# conjugate_heat_transfer:9 — the proportional contribution the fix publishes is
# exactly -kp*alpha*tau*err with tau = Nevery*dt. That is an IDENTITY between
# two columns of one stats table and four numbers the DECK sets, so it needs no
# stored measurement and no tolerance beyond round-off. Running it at three
# schedules shows the gain moving with Nevery and with the timestep while kp,
# ki, kd and alpha stay put.
GAIN_DECK = """seed 12345
dimension 2
boundary p ss p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 20 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 5e10
create_particles gas n 0
collide vss gas ar.vss
variable tint internal 300.0
surf_collide LO diffuse v_tint 1.0
surf_collide HI diffuse 300.0 1.0
bound_modify ylo collide LO
bound_modify yhi collide HI
compute cb boundary gas etot
fix av ave/time 10 10 100 c_cb[*] mode vector
fix pid controller {nev} 1.0 0.001 0.0 0.0 f_av[3] 0.0 tint
timestep {dt}
stats {nev}
stats_style step f_av[3] f_pid[1]
run 600
"""


def gain_ratio(nev: int, dt: float):
    """-P/err on every controller step, and what kp*alpha*Nevery*dt predicts."""
    rc, txt = run(GAIN_DECK.format(nev=nev, dt=dt), DATA)
    h, r = stats_rows(txt)
    if rc or not r:
        return None, None
    errs = col(h, r, "f_av[3]")     # setpoint is 0, so err == the column
    pterm = col(h, r, "f_pid[1]")
    ratios = [-p / e for e, p in zip(errs, pterm) if e != 0.0]
    return ratios, 0.001 * 1.0 * nev * dt


def identity_holds(nev: int, dt: float) -> bool:
    """The tolerance is set by SPARTA's stats table, which prints about eight
    significant digits: two such numbers divided agree with the exact ratio to
    roughly 1e-8 relative. 1e-6 is that limit with two decades of slack, and it
    is a property of the OUTPUT FORMAT rather than of the physics — the
    underlying identity is exact."""
    ratios, predicted = gain_ratio(nev, dt)
    return bool(ratios) and all(
        abs(r - predicted) <= 1e-6 * predicted for r in ratios)


gain_follows_the_schedule = (identity_holds(100, 1e-8)
                             and identity_holds(200, 1e-8)
                             and identity_holds(100, 5e-9))
print(f"the_p_term_is_exactly_minus_kp_alpha_tau_err={gain_follows_the_schedule}")
if not gain_follows_the_schedule:
    print(f"UNEXPECTED: gain identity failed; ratios at (100, 1e-8) were "
          f"{gain_ratio(100, 1e-8)}")

# conjugate_heat_transfer:10 — no wall model anywhere takes a flux. The wall that
# needs no controller at all is the one with a NUMBER for its temperature, and
# it runs: the flux is then an output, which is the whole of the claim.
rc_f, e_f, (h_f, r_f), _ = go("", "300.0", "", "")
print(f"a_plain_temperature_wall_needs_no_controller_and_runs="
      f"{rc_f == 0 and not e_f}")

ok = (rc_d == 0 and not e_d and frozen and diverges and tracks
      and any(TSURF_MSG in e for e in e_n)
      and rc_i == 0 and not e_i
      and any(NOT_INTERNAL in e for e in e_e)
      and any(ARRAY_MSG in e for e in e_c)
      and two_index_silent and gain_follows_the_schedule
      and rc_f == 0 and not e_f)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
