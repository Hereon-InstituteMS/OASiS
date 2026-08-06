"""Tier-2: what SPARTA checks about a 'collide' line, and WHEN.

  collision_relaxation:0  'collide vss <mixID>' with no filename is a hard
                          error; a .vss file that does not list one of your
                          species is caught only when the collision model is
                          set up.
  collision_relaxation:1  the mixture on the collide line must contain EVERY
                          declared species — a subset is REJECTED, not silently
                          ignored — and the check happens at the start of the
                          FIRST RUN. The second half is the one that matters and
                          it is verified by omission: the identical deck with
                          the 'run' line deleted exits 0 and reports nothing.
  collision_relaxation:3  a collide_modify typo aborts with a generic message
                          that does not name the offending keyword.

Also executes the three hard-error forms the universal 'no collide command'
entry lists, which is where an agent's guard would be built: collide_modify,
fix vibmode and react tce each abort when no collide style is defined.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, run, skip_if_unavailable  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss", "air.species", "air.vss",
                           "air.tce")

BOX = """seed 12345
dimension 2
boundary rr rr p
create_box 0 1e-4 0 1e-4 -0.5 0.5
create_grid 10 10 1
global nrho 1e21 fnum 1e12
"""
RUN = "timestep 1e-9\nstats 50\nstats_style step np ncoll\nrun 100\n"
NO_RUN = "timestep 1e-9\nstats 50\nstats_style step np ncoll\n"

AR = "species ar.species Ar\nmixture gas Ar temp 273.15\ncreate_particles gas n 0\n"
AIR2 = ("species air.species N2 O2\nmixture sub N2 temp 273.15\n"
        "mixture both N2 O2 temp 273.15\ncreate_particles both n 0\n")

CASES = {
    "collide_vss_without_a_filename":
        BOX + AR + "collide vss gas\n" + RUN,
    "vss_file_missing_one_of_your_species":
        BOX + AR + "collide vss gas air.vss\n" + RUN,
    "collide_mixture_is_a_subset_with_a_run":
        BOX + AIR2 + "collide vss sub air.vss\n" + RUN,
    "collide_mixture_is_a_subset_without_a_run":
        BOX + AIR2 + "collide vss sub air.vss\n" + NO_RUN,
    "collide_mixture_id_mistyped":
        BOX + AR + "collide vss nosuch ar.vss\n" + RUN,
    "collide_modify_keyword_typo":
        BOX + AR + "collide vss gas ar.vss\ncollide_modify vibrat smooth\n" + RUN,
    # the three hard-error forms of the universal no-collide entry
    "collide_modify_with_no_collide_style":
        BOX + AR + "collide_modify vibrate smooth\n" + RUN,
    "fix_vibmode_with_no_collide_style":
        BOX + "species air.species N2\nmixture gas N2 temp 273.15\n"
        "create_particles gas n 0\nfix v vibmode\n" + RUN,
    "react_tce_with_no_collide_vss":
        BOX + "species air.species N2 O2 NO N O\nmixture gas N2 O2 temp 273.15\n"
        "create_particles gas n 0\nreact tce air.tce\n" + RUN,
}

QUOTED = {
    "collide_vss_without_a_filename":
        "ERROR: Illegal collide command (../collide_vss.cpp:47)",
    "vss_file_missing_one_of_your_species":
        "ERROR on proc 0: Species Ar did not appear in VSS parameter file "
        "(../collide_vss.cpp:924)",
    "collide_mixture_is_a_subset_with_a_run":
        "ERROR: Collision mixture does not contain all species "
        "(../collide.cpp:159)",
    "collide_mixture_id_mistyped":
        "ERROR: Collision mixture does not exist (../collide.cpp:155)",
    "collide_modify_keyword_typo":
        "ERROR: Illegal collide_modify command (../collide.cpp:1727)",
    "collide_modify_with_no_collide_style":
        "ERROR: Cannot use collide_modify with no collisions defined "
        "(../input.cpp:1425)",
    "fix_vibmode_with_no_collide_style":
        "ERROR: Cannot use fix vibmode without collide style defined "
        "(../fix_vibmode.cpp:50)",
    "react_tce_with_no_collide_vss":
        "ERROR: React tce can only be used with collide vss "
        "(../react_tce.cpp:40)",
}

res = {}
for name, deck in CASES.items():
    rc, txt = run(deck, DATA)
    res[name] = (rc, errors(txt))
    print(f"{name}_rc={rc}")
    if errors(txt):
        print(f"{name}_message={errors(txt)[0]}")

quoted_ok = True
for name, msg in QUOTED.items():
    if msg not in res[name][1]:
        quoted_ok = False
        print(f"UNEXPECTED: {name} printed {res[name][1][:1]} not {msg!r}")

deferred = res["collide_mixture_is_a_subset_without_a_run"]
typo_msg = res["collide_modify_keyword_typo"][1][:1]
names_the_keyword = any("vibrat" in m for m in typo_msg)

print(f"every_quoted_collide_message_is_verbatim={quoted_ok}")
print(f"subset_mixture_is_rejected_at_the_first_run="
      f"{res['collide_mixture_is_a_subset_with_a_run'][0] != 0}")
print(f"same_deck_without_a_run_reports_nothing_and_exits_zero="
      f"{deferred[0] == 0 and not deferred[1]}")
print(f"collide_modify_typo_message_does_not_name_the_keyword="
      f"{not names_the_keyword}")
print(f"all_three_no_collide_hard_error_forms_abort="
      f"{all(res[k][0] != 0 for k in ('collide_modify_with_no_collide_style', 'fix_vibmode_with_no_collide_style', 'react_tce_with_no_collide_vss'))}")

ok = (quoted_ok
      and res["collide_mixture_is_a_subset_with_a_run"][0] != 0
      and deferred[0] == 0 and not deferred[1]
      and not names_the_keyword)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
