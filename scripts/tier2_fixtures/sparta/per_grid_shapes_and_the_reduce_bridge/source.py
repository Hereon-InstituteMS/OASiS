"""Tier-2: `compute property/grid` breaks the per-grid bracket rule, and
`compute reduce` is the only road from a spatial compute to the stats table.

  rarefied_flow:9  property/grid with ONE attribute is a per-grid VECTOR (c_ID,
                    no bracket) and only with two or more an ARRAY (c_ID[i]) —
                    the reverse of compute grid / compute surf, which are always
                    arrays. It also carries GEOMETRY ONLY.
  rarefied_flow:10  a per-particle / per-grid / per-surf compute cannot be named
                    in stats_style; compute reduce is the bridge, and its
                    'replace' keyword needs min/max and two distinct columns.

The two shape mistakes have OPPOSITE messages, which is the practical point: an
agent that reads the error can tell which way round it got the rule, and the
same deck run four ways here produces all four.

Nothing numeric is asserted. The one place a value is read at all — the summed
cell volume — is compared against the box volume computed from the deck's own
create_box line, so the assertion is an identity the deck states, not a number
this fixture remembers.

Mutation control: T2_MUTATE=1 gives 'compute property/grid' a SECOND attribute
in the deck whose whole point is that one attribute makes a vector. That single
edit turns the compute into an array, so the bare c_p that worked now fails and
the bracketed c_p[1] that failed now works: `one_attribute_is_a_vector` and
`bracketing_a_one_attribute_property_grid_fails` both go False. Nothing else in
any deck changes — same box, grid, seed, species, run length.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import col, errors, run, skip_if_unavailable, stats_rows  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

LX = 1.0e-3          # the deck's own box edge, used for the volume identity
NCELL = 10

DECK = """seed 12345
dimension 2
boundary p p p
create_box 0 {lx} 0 {lx} -0.5 0.5
create_grid {n} {n} 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e11
create_particles gas n 0
collide vss gas ar.vss
timestep 1e-8
{body}
stats 50
stats_style step {cols}
run 50
"""


def go(body: str, cols: str):
    rc, txt = run(DECK.format(lx=LX, n=NCELL, body=body, cols=cols), DATA)
    return rc, errors(txt), txt


# Under mutation the ONE-attribute property/grid gains a second attribute, so
# it is no longer a vector. Everything else is held fixed.
ONE_ATTR = "vol xc" if MUTATE else "vol"
if MUTATE:
    print("mutation=property_grid_given_a_second_attribute_so_it_is_no_longer_a_vector")

rc_v, e_v, txt_v = go(f"compute p property/grid all {ONE_ATTR}\n"
                      "compute r reduce sum c_p", "c_r")
rc_vb, e_vb, _ = go(f"compute p property/grid all {ONE_ATTR}\n"
                    "compute r reduce sum c_p[1]", "c_r")
rc_a, e_a, _ = go("compute p property/grid all vol xc\n"
                  "compute r reduce sum c_p", "c_r")
rc_ab, e_ab, txt_ab = go("compute p property/grid all vol xc\n"
                         "compute r reduce sum c_p[1] c_p[2]", "c_r[1] c_r[2]")

ARRAY_MSG = ("ERROR: Compute reduce compute does not calculate a per-grid "
             "array (../compute_reduce.cpp:232)")
VECTOR_MSG = ("ERROR: Compute reduce compute does not calculate a per-grid "
              "vector (../compute_reduce.cpp:229)")

one_is_vector = rc_v == 0 and not e_v
print(f"one_attribute_is_a_vector={one_is_vector}")
bracket_fails = any(ARRAY_MSG in e for e in e_vb)
print(f"bracketing_a_one_attribute_property_grid_fails={bracket_fails}")
if not bracket_fails and not MUTATE:
    print(f"UNEXPECTED: bracketed one-attribute case printed {e_vb}")
two_bare_fails = any(VECTOR_MSG in e for e in e_a)
print(f"two_attributes_unbracketed_fails={two_bare_fails}")
two_bracketed_works = rc_ab == 0 and not e_ab
print(f"two_attributes_bracketed_works={two_bracketed_works}")
print("the_two_shape_mistakes_have_opposite_messages="
      f"{bracket_fails and two_bare_fails and ARRAY_MSG != VECTOR_MSG}")

# compute grid is ALWAYS an array — the same bare reference that is legal for a
# one-attribute property/grid is rejected for a one-value compute grid.
rc_g, e_g, _ = go("compute g grid all all nrho\ncompute r reduce sum c_g", "c_r")
grid_is_always_array = any(VECTOR_MSG in e for e in e_g)
print(f"compute_grid_is_always_an_array={grid_is_always_array}")

# Geometry only. A flow quantity and a 2d-invalid field each have their own
# message, and neither is the generic 'Illegal' one.
rc_n, e_n, _ = go("compute p property/grid all nrho", "step")
FLOW_MSG = ("ERROR: Invalid keyword in compute property/grid command "
            "(../compute_property_grid.cpp:84)")
print(f"property_grid_rejects_a_flow_quantity={any(FLOW_MSG in e for e in e_n)}")
rc_z, e_z, _ = go("compute p property/grid all zc", "step")
Z2D_MSG = ("ERROR: Invalid compute property/grid field for 2d simulation "
           "(../compute_property_grid.cpp:54)")
print(f"property_grid_rejects_zc_in_2d={any(Z2D_MSG in e for e in e_z)}")

# The one number read at all: summed cell volume equals the box volume the deck
# itself declares (lx*lx per one metre of depth in 2d). An identity, not a
# remembered value.
volume_identity = False
if rc_ab == 0:
    header, rows = stats_rows(txt_ab)
    if header and rows:
        total = col(header, rows, "c_r[1]")[0]
        volume_identity = abs(total - LX * LX) <= 1e-12 * LX * LX
print(f"summed_cell_volume_is_the_box_volume_per_metre_depth={volume_identity}")

# rarefied_flow:10 — a per-particle compute cannot be printed directly, and
# 'replace' needs min/max plus two distinct columns.
rc_p, e_p, _ = go("compute kp ke/particle", "c_kp")
STATS_MSG = "ERROR: Stats compute does not compute scalar (../stats.cpp:678)"
per_particle_blocked = any(STATS_MSG in e for e in e_p)
print(f"a_per_particle_compute_cannot_go_into_stats_style={per_particle_blocked}")
rc_pr, e_pr, _ = go("compute kp ke/particle\ncompute r reduce ave c_kp", "c_r")
reduce_bridges = rc_pr == 0 and not e_pr
print(f"compute_reduce_bridges_it={reduce_bridges}")
rc_rp, e_rp, _ = go("compute g grid all all nrho\n"
                    "compute r reduce max c_g[1] replace 1 1", "c_r")
REPLACE_MSG = ("ERROR: Illegal compute reduce command "
               "(../compute_reduce.cpp:168)")
print(f"replace_needs_two_distinct_columns={any(REPLACE_MSG in e for e in e_rp)}")

ok = (one_is_vector and bracket_fails and two_bare_fails
      and two_bracketed_works and grid_is_always_array
      and any(FLOW_MSG in e for e in e_n) and any(Z2D_MSG in e for e in e_z)
      and volume_identity and per_particle_blocked and reduce_bridges
      and any(REPLACE_MSG in e for e in e_rp))
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
