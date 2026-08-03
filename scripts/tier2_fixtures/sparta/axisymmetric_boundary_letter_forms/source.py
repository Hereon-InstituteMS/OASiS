"""Tier-2: how axisymmetry is actually switched on in SPARTA.

There is no `global ... axisymmetric` keyword. The style is a BOUNDARY letter
`a`, legal only on the LOWER y face, so the single-letter form `boundary p a p`
(which sets both y faces) is always an error and the two-letter form
`boundary o ar p` is required. ylo must be exactly 0.0, yhi may not be
periodic, and `global ... weight cell radius` needs both an axisymmetric model
and an existing grid.

The 2d-only restriction is checked at RUN SETUP, not at parse time: a 3d deck
carrying `boundary o ar p` that never reaches a `run` exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, run, skip_if_unavailable  # noqa: E402

skip_if_unavailable()

CASES = {
    "global_axisymmetric_keyword": (
        "seed 1\ndimension 2\nglobal axisymmetric yes\n"
        "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 4 4 1\nrun 1\n",
        "Illegal global command (../update.cpp:1805)"),
    "single_letter_a_sets_both_faces": (
        "seed 1\ndimension 2\nboundary p a p\n"
        "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 4 4 1\nrun 1\n",
        "Only ylo boundary can be axi-symmetric (../domain.cpp:176)"),
    "a_on_yhi": (
        "seed 1\ndimension 2\nboundary o ra p\n"
        "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 4 4 1\nrun 1\n",
        "Only ylo boundary can be axi-symmetric (../domain.cpp:176)"),
    "ylo_not_zero": (
        "seed 1\ndimension 2\nboundary o ar p\n"
        "create_box 0 1 0.1 1 -0.5 0.5\ncreate_grid 4 4 1\nrun 1\n",
        "Box ylo must be 0.0 for axi-symmetric model (../create_box.cpp:56)"),
    "yhi_periodic": (
        "seed 1\ndimension 2\nboundary o ap p\n"
        "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 4 4 1\nrun 1\n",
        "Y cannot be periodic for axi-symmetric (../domain.cpp:181)"),
    "three_d_with_run": (
        "seed 1\ndimension 3\nboundary o ar p\n"
        "create_box 0 1 0 1 0 1\ncreate_grid 4 4 4\nrun 1\n",
        "Axi-symmetry only allowed for 2d simulation (../domain.cpp:80)"),
    "weight_radius_without_axisymmetry": (
        "seed 1\ndimension 2\nboundary o ro p\n"
        "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 4 4 1\n"
        "global weight cell radius\nrun 1\n",
        "Cannot use weight cell radius unless axisymmetric (../grid.cpp:1997)"),
    "weight_radius_before_grid": (
        "seed 1\ndimension 2\nboundary o ar p\n"
        "create_box 0 1 0 1 -0.5 0.5\nglobal weight cell radius\n"
        "create_grid 4 4 1\nrun 1\n",
        "Cannot weight cells before grid is defined (../grid.cpp:1983)"),
}

ACCEPTED = {
    "two_letter_ar": "seed 1\ndimension 2\nboundary o ar p\n"
                     "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 4 4 1\n"
                     "global weight cell radius\nrun 1\n",
    "two_letter_ao": "seed 1\ndimension 2\nboundary p ao p\n"
                     "create_box 0 1 0 1 -0.5 0.5\ncreate_grid 4 4 1\nrun 1\n",
}

matched = []
for name, (deck, expect) in CASES.items():
    rc, txt = run(deck, timeout=120)
    errs = errors(txt)
    hit = rc != 0 and any(expect in e for e in errs)
    matched.append(hit)
    print(f"{name}: rc={rc} matched={hit} err={errs[:1]}")

accepted = []
for name, deck in ACCEPTED.items():
    rc, txt = run(deck, timeout=120)
    accepted.append(rc == 0)
    print(f"{name}: rc={rc} accepted={rc == 0}")

# the 2d-only check is deferred: no 'run' => no error
rc_norun, txt_norun = run(
    "seed 1\ndimension 3\nboundary o ar p\n"
    "create_box 0 1 0 1 0 1\ncreate_grid 4 4 4\n", timeout=120)
print(f"three_d_without_run_rc={rc_norun} errors={errors(txt_norun)[:1]}")

print(f"all_expected_errors_matched={all(matched)}")
print(f"both_two_letter_forms_accepted={all(accepted)}")
print(f"two_d_check_is_deferred_to_run_setup={rc_norun == 0}")

if not (all(matched) and all(accepted) and rc_norun == 0):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
