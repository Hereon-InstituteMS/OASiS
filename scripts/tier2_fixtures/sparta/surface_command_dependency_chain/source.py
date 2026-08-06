"""Tier-2: the SPARTA surface commands have a hard dependency order, and every
step of it aborts rather than running on with an unbound surface.

Three claims are executed here because they share one deck and one surface file:

  surface_interaction:2  every element must be bound to a collision model
                         before the first run, and a PARTIAL binding is caught
                         too — with the count of elements STILL UNBOUND in the
                         message, not the total;
  surface_interaction:3  read_surf needs the grid; surf_modify needs both the
                         surfaces and the surf_collide model; a mistyped model
                         ID has its own message; surf_collide itself is NOT
                         tied to read_surf and may be declared earlier;
  surface_interaction:9  a BOX face is only a surface if declared 's', it then
                         needs bound_modify, and bound_modify on a face that is
                         'o' or 'p' is rejected.

The interesting negative result is in the middle one: the knowledge says the
partial-binding case is caught, and it is — every wrong ordering here exits
NON-ZERO with a named message. Nothing in this group fails silently, which is
what makes it cheap to depend on.

Every expected string below was captured from this deck on the installed build;
none is composed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, run, skip_if_unavailable  # noqa: E402

DATA = skip_if_unavailable("ar.species", "data.circle")

HEAD = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary {bnd}
create_box 0 10 0 10 -0.5 0.5
"""
GRID = "create_grid 10 10 1\n"
GAS = """global nrho 1e20 fnum 1e17
species ar.species Ar
mixture gas Ar vstream 0 0 0 temp 273.15
"""
TAIL = "timestep 1e-6\nstats 10\nstats_style step np nscoll\nrun 20\n"
WALL = "surf_collide cw diffuse 300 1.0\n"

CASES = {
    # ---- surface_interaction:3 — the ordering chain -----------------------
    "read_surf_before_grid":
        HEAD.format(bnd="o o p") + GAS + "read_surf data.circle\n" + GRID + TAIL,
    "surf_modify_before_surfs":
        HEAD.format(bnd="o o p") + GRID + GAS + WALL
        + "surf_modify all collide cw\n" + TAIL,
    "surf_modify_bad_sc_id":
        HEAD.format(bnd="o o p") + GRID + GAS + "read_surf data.circle\n" + WALL
        + "surf_modify all collide typo\n" + TAIL,
    "surf_modify_bad_sr_id":
        HEAD.format(bnd="o o p") + GRID + GAS + "read_surf data.circle\n" + WALL
        + "surf_modify all collide cw react typo\n" + TAIL,
    # ---- surface_interaction:2 — binding is mandatory, partial counts -----
    "no_binding_at_all":
        HEAD.format(bnd="o o p") + GRID + GAS + "read_surf data.circle\n" + TAIL,
    "partial_group_binding":
        HEAD.format(bnd="o o p") + GRID + GAS + "read_surf data.circle\n" + WALL
        + "group left surf id <= 25\nsurf_modify left collide cw\n" + TAIL,
    # ---- surface_interaction:9 — a box face is not a surface unless 's' ---
    "s_face_declared_but_unbound":
        HEAD.format(bnd="o s p") + GRID + GAS + TAIL,
    "bound_modify_on_outflow_face":
        HEAD.format(bnd="o o p") + GRID + GAS + WALL
        + "bound_modify yhi collide cw\n" + TAIL,
    "bound_modify_on_periodic_face":
        HEAD.format(bnd="p o p") + GRID + GAS + WALL
        + "bound_modify xlo collide cw\n" + TAIL,
}

# The forms the knowledge says are CORRECT. If these do not run, the entry is
# telling an agent to write a deck that does not work.
GOOD = {
    "correct_order":
        HEAD.format(bnd="o o p") + GRID + GAS + "read_surf data.circle\n" + WALL
        + "surf_modify all collide cw\n" + TAIL,
    "surf_collide_declared_before_read_surf":
        HEAD.format(bnd="o o p") + GRID + GAS + WALL + "read_surf data.circle\n"
        + "surf_modify all collide cw\n" + TAIL,
    "both_groups_bound":
        HEAD.format(bnd="o o p") + GRID + GAS + "read_surf data.circle\n" + WALL
        + "group left surf id <= 25\ngroup right surf id > 25\n"
        + "surf_modify left collide cw\nsurf_modify right collide cw\n" + TAIL,
    "s_face_bound_with_bound_modify":
        HEAD.format(bnd="o s p") + GRID + GAS + WALL
        + "bound_modify ylo collide cw\nbound_modify yhi collide cw\n" + TAIL,
}

# Message text the knowledge quotes, keyed by the case that must produce it.
# Compared as a whole line INCLUDING the (file:line) suffix SPARTA appends.
EXPECT = {
    "read_surf_before_grid":
        "ERROR: Cannot read_surf before grid is defined (../read_surf.cpp:73)",
    "surf_modify_before_surfs":
        "ERROR: Surf_modify when surfs do not yet exist (../surf.cpp:227)",
    "surf_modify_bad_sc_id":
        "ERROR: Could not find surf_modify sc-ID (../surf.cpp:230)",
    "surf_modify_bad_sr_id":
        "ERROR: Could not find surf_modify sr-ID (../surf.cpp:260)",
    "no_binding_at_all":
        "ERROR: 50 surface elements not assigned to a collision model "
        "(../surf.cpp:343)",
    "partial_group_binding":
        "ERROR: 25 surface elements not assigned to a collision model "
        "(../surf.cpp:343)",
    "s_face_declared_but_unbound":
        "ERROR: Box boundary not assigned a surf_collide ID (../domain.cpp:100)",
    "bound_modify_on_outflow_face":
        "ERROR: Bound_modify surf requires boundary be a surface "
        "(../domain.cpp:253)",
    "bound_modify_on_periodic_face":
        "ERROR: Bound_modify surf requires boundary be a surface "
        "(../domain.cpp:253)",
}

wrong, right = {}, {}
for name, deck in CASES.items():
    rc, txt = run(deck, DATA)
    errs = errors(txt)
    got = EXPECT[name] in errs
    wrong[name] = (rc, got, errs[:1])
    print(f"{name}_rc={rc} quoted_message_matched={got}")
    if not got:
        print(f"UNEXPECTED: {name} printed {errs[:1]} not {EXPECT[name]!r}")

for name, deck in GOOD.items():
    rc, txt = run(deck, DATA)
    right[name] = rc
    print(f"{name}_rc={rc}")

# --- the four assertions this fixture exists to make ----------------------
all_abort = all(v[0] != 0 for v in wrong.values())
all_quoted = all(v[1] for v in wrong.values())
partial_names_unbound_count = wrong["partial_group_binding"][1]
good_all_run = all(rc == 0 for rc in right.values())

print(f"every_wrong_ordering_aborts_nonzero={all_abort}")
print(f"every_quoted_message_is_verbatim={all_quoted}")
print(f"partial_binding_reports_the_unbound_count_not_the_total="
      f"{partial_names_unbound_count}")
print(f"surf_collide_may_precede_read_surf="
      f"{right['surf_collide_declared_before_read_surf'] == 0}")
print(f"every_form_the_entry_calls_correct_runs={good_all_run}")

if not (all_abort and all_quoted and good_all_run):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
