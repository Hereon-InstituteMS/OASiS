"""Tier-2: the three ways a SPARTA radiative-equilibrium wall is declared wrong.

  conjugate_heat_transfer:0  fix surf/temp CREATES the custom per-surf
                             attribute, so the surf_collide line consuming it
                             as s_<name> must come afterwards. The intuitive
                             order (wall model first) aborts — with EXACTLY the
                             message you get when the fix is missing entirely,
                             so the diagnostic cannot tell the two apart. Both
                             halves of that are checked here, because the claim
                             is about the ambiguity, not just the abort.
  conjugate_heat_transfer:1  the <source> argument is BRACKET-driven, not
                             compute-versus-fix. A bare compute fails, and so
                             does the SPARTA doc page's own 'c_1' example;
                             'c_<ID>[*]' fails identically because the wildcard
                             parses as index 0; 'c_<ID>[1]' runs. A fix ave/surf
                             with one input is a VECTOR, so f_<ID>[1] aborts.
  conjugate_heat_transfer:2  emissivity is strictly inside (0, 1]. Zero is
                             rejected, not read as 'no radiation'; 1.0 exactly
                             is accepted.

The load-bearing check is on cht:1: an entry that tells an agent the DOC PAGE is
wrong has to be able to show it, so both the doc form and the working form run
here side by side.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, run, skip_if_unavailable  # noqa: E402

DATA = skip_if_unavailable("ar.species", "ar.vss", "data.circle")

DECK = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o o p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
global nrho 1e20 fnum 1e17
species ar.species Ar
mixture gas Ar vstream 100 0 0 temp 273.15
read_surf data.circle
{body}create_particles gas n 0
collide vss gas ar.vss
timestep 1e-6
stats 100
stats_style step np nscoll
run 200
"""

COMPUTE = "compute q surf all all etot\n"
AVE = "fix aq ave/surf all 10 10 100 c_q[1]\n"
WALL = "surf_collide cw diffuse s_tw 1.0\nsurf_modify all collide cw\n"

CASES = {
    # cht:0 — ordering, and the ambiguity of the message
    "correct_order_fix_then_surf_collide":
        COMPUTE + AVE + "fix ft surf/temp all 100 f_aq 300 0.9 tw\n" + WALL,
    "intuitive_wrong_order_surf_collide_first":
        COMPUTE + AVE + WALL + "fix ft surf/temp all 100 f_aq 300 0.9 tw\n",
    "fix_surf_temp_missing_entirely": WALL,
    # cht:1 — brackets
    "bare_compute_source_the_doc_page_example":
        COMPUTE + "fix ft surf/temp all 100 c_q 300 0.9 tw\n" + WALL,
    "compute_wildcard_index":
        COMPUTE + "fix ft surf/temp all 100 c_q[*] 300 0.9 tw\n" + WALL,
    "indexed_compute_source":
        COMPUTE + "fix ft surf/temp all 100 c_q[1] 300 0.9 tw\n" + WALL,
    "single_value_fix_over_indexed":
        COMPUTE + AVE + "fix ft surf/temp all 100 f_aq[1] 300 0.9 tw\n" + WALL,
    # cht:2 — emissivity range
    "emissivity_zero":
        COMPUTE + AVE + "fix ft surf/temp all 100 f_aq 300 0.0 tw\n" + WALL,
    "emissivity_above_one":
        COMPUTE + AVE + "fix ft surf/temp all 100 f_aq 300 1.5 tw\n" + WALL,
    "emissivity_one_exactly":
        COMPUTE + AVE + "fix ft surf/temp all 100 f_aq 300 1.0 tw\n" + WALL,
}

MUST_RUN = {"correct_order_fix_then_surf_collide", "indexed_compute_source",
            "emissivity_one_exactly"}

QUOTED = {
    "intuitive_wrong_order_surf_collide_first":
        "ERROR: Surf_collide tsurf could not find custom attribute "
        "(../surf_collide.cpp:141)",
    "fix_surf_temp_missing_entirely":
        "ERROR: Surf_collide tsurf could not find custom attribute "
        "(../surf_collide.cpp:141)",
    "bare_compute_source_the_doc_page_example":
        "ERROR: Fix surf/temp compute does not compute per-surf vector "
        "(../fix_surf_temp.cpp:82)",
    "compute_wildcard_index":
        "ERROR: Fix surf/temp compute does not compute per-surf vector "
        "(../fix_surf_temp.cpp:82)",
    "single_value_fix_over_indexed":
        "ERROR: Fix surf/temp fix does not compute per-surf array "
        "(../fix_surf_temp.cpp:112)",
    "emissivity_zero":
        "ERROR: Fix surf/temp emissivity must be > 0.0 and <= 1 "
        "(../fix_surf_temp.cpp:125)",
    "emissivity_above_one":
        "ERROR: Fix surf/temp emissivity must be > 0.0 and <= 1 "
        "(../fix_surf_temp.cpp:125)",
}

res = {}
for name, body in CASES.items():
    rc, txt = run(DECK.format(body=body), DATA)
    res[name] = (rc, errors(txt))
    print(f"{name}_rc={rc}")
    if errors(txt):
        print(f"{name}_message={errors(txt)[0]}")

quoted_ok = True
for name, msg in QUOTED.items():
    if msg not in res[name][1]:
        quoted_ok = False
        print(f"UNEXPECTED: {name} printed {res[name][1][:1]} not {msg!r}")

good_run = all(res[n][0] == 0 for n in MUST_RUN)
same_msg = (res["intuitive_wrong_order_surf_collide_first"][1][:1]
            == res["fix_surf_temp_missing_entirely"][1][:1])
doc_form_fails = res["bare_compute_source_the_doc_page_example"][0] != 0
wildcard_same_as_bare = (res["compute_wildcard_index"][1][:1]
                         == res["bare_compute_source_the_doc_page_example"][1][:1])

print(f"every_quoted_fix_surf_temp_message_is_verbatim={quoted_ok}")
print(f"forms_the_entry_calls_correct_all_run={good_run}")
print(f"wrong_order_and_missing_fix_give_the_same_message={same_msg}")
print(f"doc_page_bare_compute_form_does_not_work={doc_form_fails}")
print(f"wildcard_index_fails_exactly_like_the_bare_form={wildcard_same_as_bare}")
print(f"emissivity_zero_is_rejected_not_read_as_no_radiation="
      f"{res['emissivity_zero'][0] != 0}")

if not (quoted_ok and good_run and same_msg and doc_form_fails
        and wildcard_same_as_bare):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
