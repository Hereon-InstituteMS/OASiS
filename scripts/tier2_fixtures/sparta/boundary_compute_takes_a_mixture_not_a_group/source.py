"""Tier-2: `compute boundary` is the one spatial compute that takes a mixture
and no group, and it is a global array.

  hypersonic_flow:6  copying the 'compute grid <group> <mixture> ...' pattern
                     consumes 'all' as the mixture and then rejects the second
                     'all' as a value; and without 'mode vector' fix ave/time
                     asks it for a scalar it cannot produce. Rows are the box
                     faces in the order xlo, xhi, ylo, yhi (only FOUR in 2d).

The row ordering is checked structurally, not against remembered numbers: in a
box periodic in x and surfaced in y, the xlo and xhi rows must tally exactly
zero for the whole run while the ylo and yhi rows must not — which is what a
row order of xlo, xhi, ylo, yhi means and what any other order would break.

Mutation control: T2_MUTATE=1 makes the x faces OUTFLOW instead of periodic, so
particles do cross them and the xlo/xhi rows stop being identically zero. The
row-order evidence then collapses and `the_first_two_rows_are_the_periodic_x_
faces` goes False. Deliberately not a change of seed or run length: the point is
that those two rows are the x faces, so changing what the x faces do is the edit
that speaks.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

XBOUND = "o" if MUTATE else "p"
if MUTATE:
    print("mutation=x_faces_made_outflow_so_they_are_no_longer_silent")

DECK = """seed 12345
dimension 2
boundary {xb} ss p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 20 1
species ar.species Ar
mixture gas Ar vstream 200.0 0 0 temp 300.0
global nrho 1e21 fnum 1e10
create_particles gas n 0
collide vss gas ar.vss
surf_collide LO diffuse 300.0 1.0
surf_collide HI diffuse 1000.0 1.0
bound_modify ylo collide LO
bound_modify yhi collide HI
compute cb boundary {args}
{consumer}
timestep 1e-8
stats 200
stats_style step np {cols}
run 400
"""


def go(args: str, consumer: str, cols: str, xb: str = XBOUND):
    rc, txt = run(DECK.format(xb=xb, args=args, consumer=consumer, cols=cols),
                  DATA)
    h, r = stats_rows(txt)
    return rc, errors(txt), (h, r)


# The group-then-mixture habit from compute grid / compute surf.
rc_g, e_g, _ = go("all all etot", "", "")
GROUP_MSG = "ERROR: Illegal compute boundary command (../compute_boundary.cpp:64)"
print(f"a_group_id_is_rejected={any(GROUP_MSG in e for e in e_g)}")

# Without mode vector the fix asks it for a scalar.
rc_s, e_s, _ = go("gas etot", "fix f ave/time 10 10 200 c_cb[*]", "f_f")
SCALAR_MSG = ("ERROR: Fix ave/time compute does not calculate a scalar "
              "(../fix_ave_time.cpp:137)")
print(f"mode_vector_is_mandatory={any(SCALAR_MSG in e for e in e_s)}")

# The working form. One compute value, so the fix is a global VECTOR indexed
# once per face.
rc_v, e_v, (h_v, r_v) = go(
    "gas etot", "fix f ave/time 10 10 200 c_cb[*] mode vector",
    "f_f[1] f_f[2] f_f[3] f_f[4]")
works = rc_v == 0 and not e_v and bool(r_v)
print(f"the_mixture_only_form_with_mode_vector_works={works}")

xlo = col(h_v, r_v, "f_f[1]") if works else []
xhi = col(h_v, r_v, "f_f[2]") if works else []
ylo = col(h_v, r_v, "f_f[3]") if works else []
yhi = col(h_v, r_v, "f_f[4]") if works else []

x_silent = bool(xlo) and all(v == 0.0 for v in xlo) and all(v == 0.0 for v in xhi)
y_alive = bool(ylo) and any(v != 0.0 for v in ylo[1:]) and any(v != 0.0 for v in yhi[1:])
print(f"the_first_two_rows_are_the_periodic_x_faces={x_silent}")
if not x_silent and not MUTATE:
    print(f"UNEXPECTED: x rows were not silent: {xlo[-1:]} {xhi[-1:]}")
print(f"the_last_two_rows_are_the_surfaced_y_faces={y_alive}")

# In 2d there are only FOUR rows: a fifth is out of range.
rc_5, e_5, _ = go("gas etot", "fix f ave/time 10 10 200 c_cb[*] mode vector",
                  "f_f[5]")
five_rejected = rc_5 != 0 and bool(e_5)
print(f"a_fifth_row_does_not_exist_in_2d={five_rejected}")
if five_rejected:
    print(f"fifth_row_message={e_5[0]}")

# With two values the fix becomes an ARRAY and needs two indices.
rc_2, e_2, (h_2, r_2) = go(
    "gas etot press", "fix f ave/time 10 10 200 c_cb[*] mode vector",
    "f_f[3][1] f_f[3][2]")
print(f"two_values_make_it_a_face_by_value_array={rc_2 == 0 and not e_2 and bool(r_2)}")

ok = (any(GROUP_MSG in e for e in e_g) and any(SCALAR_MSG in e for e in e_s)
      and works and x_silent and y_alive and five_rejected
      and rc_2 == 0 and not e_2)
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
