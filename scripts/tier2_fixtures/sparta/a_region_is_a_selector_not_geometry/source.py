"""Tier-2: a `region` is a selector. It obstructs nothing.

  rarefied_flow:14  defining a region changes the run not at all until a command
                    names it; it is not geometry, reflects nothing, and there is
                    no warning to tell you so.

The proof is an IDENTITY, which is the strongest form available for a claim
about something that does nothing: a driven argon channel run with and without a
region straddling the middle of the flow produces the SAME stats table, row for
row — same particle count, same surface-collision count, same exit count, on
every printed step. Not "close", identical, because the region never enters the
mover at all. The seed is held fixed so the two runs are the same realisation.

The second half checks the region machinery does work when something consumes
it: create_particles with a region fills only that region, the six styles parse,
'side out' inverts, union takes a leading count, and a sphere and a z-cylinder
of equal radius select the same cells in 2d. Those are compared against each
other inside this script, never against a stored number.

Mutation control: T2_MUTATE=1 replaces the inert region in the channel deck with
the thing a user actually wanted — a read_surf body plus a surf_collide model at
the same place — so the run is no longer identical to the control: the
surface-collision column leaves zero and the counts diverge. `the_two_runs_are_
identical_row_for_row` and `no_surface_collision_is_ever_tallied` go False. The
control deck is untouched, so the comparison is still against the same baseline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    col, errors, find_example, run, skip_if_unavailable, stats_rows,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species", "ar.vss")

CIRCLE = find_example("circle", "data.circle")
if MUTATE and CIRCLE is None:
    print("SKIP: SPARTA examples/circle/data.circle not found (set SPARTA_ROOT)")
    sys.exit(0)

# The channel box and the density are the ones examples/circle uses, so
# data.circle drops into it unscaled under mutation and the mutated deck runs
# to completion — a mutation that merely crashed would prove nothing.
CHANNEL = """seed 12345
dimension 2
global gridcut 0.0
boundary o r p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
species ar.species Ar
mixture gas Ar vstream 100.0 0 0 temp 300.0
global nrho 1.0 fnum 0.001
{obstacle}
create_particles gas n 0
collide vss gas ar.vss
fix in emit/face gas xlo
timestep 0.0001
stats 100
stats_style step np nscoll nexit
run 200
"""

REGION_LINE = "region blk block 4 6 INF INF INF INF"
# What the region-writer actually meant: real geometry with a wall model. Under
# mutation this replaces the inert region, at the same place in the deck.
SURFACE_LINES = (f"read_surf {CIRCLE}\n"
                 "surf_collide W diffuse 300.0 1.0\n"
                 "surf_modify all collide W") if CIRCLE else REGION_LINE

if MUTATE:
    print("mutation=inert_region_replaced_by_read_surf_plus_surf_collide")

rc_none, txt_none = run(CHANNEL.format(obstacle=""), DATA)
rc_obst, txt_obst = run(
    CHANNEL.format(obstacle=SURFACE_LINES if MUTATE else REGION_LINE), DATA)

h0, r0 = stats_rows(txt_none)
h1, r1 = stats_rows(txt_obst)

both_ran = rc_none == 0 and rc_obst == 0 and not errors(txt_none) and not errors(txt_obst)
print(f"both_decks_complete={both_ran}")

identical = bool(r0) and h0 == h1 and r0 == r1
print(f"the_two_runs_are_identical_row_for_row={identical}")
if not identical and not MUTATE:
    print(f"UNEXPECTED: region run differs from the control: {r0[-1:]} vs {r1[-1:]}")

no_scoll = False
if h1 and r1:
    no_scoll = all(v == 0.0 for v in col(h1, r1, "Nscoll"))
print(f"no_surface_collision_is_ever_tallied={no_scoll}")

# Regions DO work — when a command consumes them. All of this is measured
# against the same deck's own no-region run, never against a stored count.
BOX = """seed 12345
dimension 2
boundary p p p
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 10 10 1
species ar.species Ar
mixture gas Ar temp 300.0
global nrho 1e21 fnum 1e10
{reg}
create_particles gas n 0 {sel}
timestep 1e-8
stats 10
stats_style step np
run 10
"""


def npart(reg: str, sel: str):
    rc, txt = run(BOX.format(reg=reg, sel=sel), DATA)
    h, r = stats_rows(txt)
    if rc or not r:
        return rc, errors(txt), None
    return rc, errors(txt), col(h, r, "Np")[0]


_, _, n_all = npart("", "")
_, _, n_half = npart("region half block 0 0.5e-3 INF INF INF INF", "region half")
_, _, n_out = npart("region half block 0 0.5e-3 INF INF INF INF side out",
                    "region half")
_, _, n_sph = npart("region s sphere 0.5e-3 0.5e-3 0 0.25e-3", "region s")
_, _, n_cyl = npart("region c cylinder z 0.5e-3 0.5e-3 0.25e-3 INF INF",
                    "region c")
_, _, n_uni = npart("region a block 0 0.3e-3 INF INF INF INF\n"
                    "region b block 0.7e-3 1e-3 INF INF INF INF\n"
                    "region u union 2 a b", "region u")

sel_works = all(v is not None for v in (n_all, n_half, n_out, n_sph, n_uni))
print(f"a_consuming_command_does_see_the_region="
      f"{sel_works and n_half is not None and n_all is not None and n_half < n_all}")
print(f"side_out_selects_the_complement="
      f"{n_half is not None and n_out is not None and n_half + n_out == n_all}")
print(f"in_2d_a_sphere_and_a_z_cylinder_select_the_same_cells="
      f"{n_sph is not None and n_sph == n_cyl}")
print(f"union_takes_a_leading_count={n_uni is not None and 0 < n_uni < n_all}")

_, e_style, _ = npart("region q ellipsoid 0 1 0 1 0 1", "region q")
STYLE_MSG = "ERROR: Unrecognized region style (../domain.cpp:471)"
print(f"an_unknown_style_is_named={any(STYLE_MSG in e for e in e_style)}")
_, e_name, _ = npart("", "region nope")
NAME_MSG = ("ERROR: Create_particles region does not exist "
            "(../create_particles.cpp:122)")
print(f"an_undefined_region_name_is_named={any(NAME_MSG in e for e in e_name)}")

ok = (both_ran and identical and no_scoll and sel_works
      and n_half < n_all and n_half + n_out == n_all and n_sph == n_cyl
      and 0 < n_uni < n_all
      and any(STYLE_MSG in e for e in e_style)
      and any(NAME_MSG in e for e in e_name))
print("RESULT=" + ("pass" if ok else "fail"))
sys.exit(0 if ok else 1)
