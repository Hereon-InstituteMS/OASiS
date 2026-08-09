"""Tier-2: the two axisymmetric claims not already covered by the boundary-letter
fixture.

  axisymmetric:3  'global weight cell radius' has TWO ordering constraints with
                  two DIFFERENT messages: the model must already be
                  axisymmetric, and the grid must already exist.
  axisymmetric:5  the 'Axisymm bad moves' counter is printed in EVERY run, so
                  its PRESENCE means nothing and only a nonzero value does. That
                  is a claim about a non-axisymmetric run too, and this fixture
                  checks it there — the line appears in a plain 'boundary o o p'
                  deck as well.

axisymmetric:5 is the more useful half. An agent told to look for a line will
find it in every log and conclude the mode is active; the honest reading is the
value.

Mutation control: T2_MUTATE=1 fixes the ORDERING in both wrong decks — the
non-axisymmetric one gets `boundary o ar p`, and the one that weighted before
create_grid weights after it — so both become the deck the fixture already runs
as the correct order, and nothing else changes. Both then exit 0 instead of
aborting, so the two verbatim messages are absent:
`both_quoted_weight_radius_messages_are_verbatim` goes False (and the two
`UNEXPECTED:` lines it prints trip forbid_in_output).

The axisymmetric:5 half CANNOT be mutation-controlled from a deck. Its claim is
about SPARTA's own output — that `Axisymm bad moves` is printed in every
completed run, axisymmetric or not — so there is no deck edit that removes the
pathology; only a change to SPARTA would. `it_appears_even_in_a_non_axisymmetric
_run` and `so_only_a_nonzero_value_carries_information` are therefore not
covered by this hook.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import errors, run, skip_if_unavailable  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

DATA = skip_if_unavailable("ar.species")

DECK = """seed 12345
dimension 2
boundary {bnd}
create_box 0 1e-4 0 1e-4 -0.5 0.5
{before_grid}create_grid 10 10 1
{after_grid}global nrho 1e21 fnum 1e12
species ar.species Ar
mixture gas Ar temp 273.15
create_particles gas n 0
timestep 1e-9
stats 50
stats_style step np
run 100
"""
WEIGHT = "global weight cell radius\n"


def probe(bnd: str, before_grid: str = "", after_grid: str = "") -> dict:
    rc, txt = run(DECK.format(bnd=bnd, before_grid=before_grid,
                              after_grid=after_grid), DATA)
    return {"rc": rc, "errors": errors(txt),
            "bad_moves": [l for l in txt.splitlines()
                          if l.startswith("Axisymm bad moves")]}


# Under mutation the pathology is REMOVED from both wrong decks: the model is
# made axisymmetric, and the weight command is moved after create_grid.
if MUTATE:
    print("mutation=both_weight_radius_decks_written_in_the_correct_order")

CASES = {
    "weight_radius_on_a_non_axisymmetric_model":
        probe("o ar p" if MUTATE else "o o p", after_grid=WEIGHT),
    "weight_radius_before_create_grid":
        probe("o ar p", before_grid="" if MUTATE else WEIGHT,
              after_grid=WEIGHT if MUTATE else ""),
    "weight_radius_in_the_correct_order":
        probe("o ar p", after_grid=WEIGHT),
    "axisymmetric_without_radial_weighting":
        probe("o ar p"),
    "plain_non_axisymmetric_run":
        probe("o o p"),
}

QUOTED = {
    "weight_radius_on_a_non_axisymmetric_model":
        "ERROR: Cannot use weight cell radius unless axisymmetric "
        "(../grid.cpp:1997)",
    "weight_radius_before_create_grid":
        "ERROR: Cannot weight cells before grid is defined (../grid.cpp:1983)",
}

for name, r in CASES.items():
    print(f"{name}_rc={r['rc']}")
    if r["errors"]:
        print(f"{name}_message={r['errors'][0]}")
    if r["bad_moves"]:
        print(f"{name}_bad_moves_line={r['bad_moves'][0]}")

quoted_ok = True
for name, msg in QUOTED.items():
    if msg not in CASES[name]["errors"]:
        quoted_ok = False
        print(f"UNEXPECTED: {name} printed {CASES[name]['errors'][:1]} "
              f"not {msg!r}")

# The two messages SPARTA ACTUALLY PRINTED, not the two the fixture typed.
# This was `QUOTED[a] != QUOTED[b]` -- both operands are the string constants
# defined thirty lines above, so the line asserted that the author typed two
# different strings and could not come out False whatever SPARTA did, on any
# version, with or without the pathology.
_got_a = CASES["weight_radius_on_a_non_axisymmetric_model"]["errors"][:1]
_got_b = CASES["weight_radius_before_create_grid"]["errors"][:1]
print(f"observed_non_axisymmetric_message={_got_a!r}")
print(f"observed_before_create_grid_message={_got_b!r}")
two_distinct = bool(_got_a and _got_b and _got_a[0] != _got_b[0])
correct_runs = CASES["weight_radius_in_the_correct_order"]["rc"] == 0
completed = ("axisymmetric_without_radial_weighting", "plain_non_axisymmetric_run",
             "weight_radius_in_the_correct_order")
line_everywhere = all(CASES[k]["bad_moves"] for k in completed)
zero_everywhere = all(
    CASES[k]["bad_moves"] and CASES[k]["bad_moves"][0].split("=")[-1].strip() == "0"
    for k in completed)

print(f"both_quoted_weight_radius_messages_are_verbatim={quoted_ok}")
print(f"the_two_ordering_constraints_have_different_messages={two_distinct}")
print(f"weight_radius_after_boundary_and_grid_runs={correct_runs}")
print(f"axisymm_bad_moves_line_appears_in_every_completed_run={line_everywhere}")
print(f"it_appears_even_in_a_non_axisymmetric_run="
      f"{bool(CASES['plain_non_axisymmetric_run']['bad_moves'])}")
print(f"so_only_a_nonzero_value_carries_information={zero_everywhere}")

if not (quoted_ok and two_distinct and correct_runs and line_everywhere
        and zero_everywhere):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
