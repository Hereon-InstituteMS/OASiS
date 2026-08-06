"""Tier-2: three setup-time traps — one loud about a missing file, one loud
about a bad reaction file, one silent about doing nothing at all.

  conjugate_heat_transfer:5  SPARTA opens every data file relative to the
                             CURRENT WORKING DIRECTORY, so a driver that starts
                             the participant in a fresh directory dies at setup
                             unless the files were staged there.
  ambipolar_plasma:4         'surf_react <ID> prob <file>' validates every
                             probability in the file at READ time.
  adaptive_grid:5            'balance_grid' on a serial build runs, prints, and
                             moves nothing — it is not a remedy for an
                             unbalanced particle load inside one rank.

The first two are executed as PAIRS: the identical deck is run once with the
file staged and once without, and once with an in-range probability and once
with one above 1.0. That is what makes the message evidence for the claim rather
than evidence that some deck somewhere fails. The third is the silent one, and
its honest Signal is a line that is printed on every call whether or not
anything happened.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _spahelp import (  # noqa: E402
    errors, find_example, require, run, skip_if_unavailable, stats_rows,
)

DATA = skip_if_unavailable("ar.species", "ar.vss", "air.species", "air.vss")

CIRCLE = find_example("circle", "data.circle")
if CIRCLE is None:                                          # pragma: no cover
    print("SKIP: SPARTA examples/circle/data.circle not found (set SPARTA_ROOT)")
    sys.exit(0)

SCRATCH = Path(tempfile.mkdtemp(prefix="spa_setup_"))
BAD = SCRATCH / "bad.surf"
BAD.write_text("# probability column out of range\n\n"
               "N --> N2\nE S 1.5 1.563e-18\n\n"
               "N --> NULL\nR S 0.5 0.0\n")
GOOD = SCRATCH / "good.surf"
GOOD.write_text("# the same file with an in-range probability\n\n"
                "N --> N2\nE S 0.5 1.563e-18\n\n"
                "N --> NULL\nR S 0.5 0.0\n")

# ------------------------------------- conjugate_heat_transfer:5, staged or not
STAGING = """seed 12345
dimension 2
global gridcut 0.0
create_box 0 1 0 1 -0.5 0.5
create_grid 5 5 1
global nrho 1e20 fnum 1e17
species ar.species Ar
mixture gas Ar temp 273.15
collide vss gas ar.vss
create_particles gas n 0
timestep 1e-6
stats 10
stats_style step np
run 10
"""

staged = run(STAGING, {"ar.species": DATA["ar.species"],
                       "ar.vss": DATA["ar.vss"]})
unstaged = run(STAGING, {})

# ------------------------------------------------- ambipolar_plasma:4, prob file
REACT = """seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o r p
create_box 0 10 0 10 -0.5 0.5
create_grid 10 10 1
global nrho 1e20 fnum 1e17
species air.species N N2
mixture air N vstream 100.0 0 0
read_surf data.circle
surf_collide 1 diffuse 300.0 0.0
surf_react sr prob {f}
surf_modify all collide 1 react sr
timestep 1e-5
stats 50
stats_style step np nsreact
run 50
"""
REACT_DATA = {"air.species": DATA["air.species"], "air.vss": DATA["air.vss"],
              "data.circle": CIRCLE}
bad_prob = run(REACT.format(f="bad.surf"),
               dict(REACT_DATA, **{"bad.surf": BAD}))
good_prob = run(REACT.format(f="good.surf"),
                dict(REACT_DATA, **{"good.surf": GOOD}))

# ----------------------------------------------------- adaptive_grid:5, balance
BALANCE = """seed 12345
dimension 2
boundary p p p
global gridcut 0.0
create_box 0 1e-3 0 1e-3 -0.5 0.5
create_grid 20 20 1
global nrho 1e22 fnum 5e11
species ar.species Ar
mixture gas Ar temp 273.15
create_particles gas n 0
balance_grid rcb cell
balance_grid rcb part
collide vss gas ar.vss
timestep 1e-7
stats 100
stats_style step np ngrid
run 100
"""
bal_rc, bal_txt = run(BALANCE, {"ar.species": DATA["ar.species"],
                                "ar.vss": DATA["ar.vss"]})
migrated = [l.strip() for l in bal_txt.splitlines()
            if "Balance grid migrated" in l]
bal_header, bal_rows = stats_rows(bal_txt)

for tag, (rc, txt) in (("species_file_staged", staged),
                       ("species_file_not_staged", unstaged),
                       ("surf_react_probability_above_one", bad_prob),
                       ("surf_react_probability_in_range", good_prob)):
    errs = errors(txt)
    print(f"{tag}_rc={rc} n_errors={len(errs)}")
    if errs:
        print(f"{tag}_message={errs[0]}")
print(f"balance_grid_rc={bal_rc} n_stats_rows={len(bal_rows)}")
for i, line in enumerate(migrated):
    print(f"balance_grid_line_{i}={line}")

QUOTED = {
    "species_file_not_staged": (
        unstaged[1],
        "ERROR on proc 0: Cannot open species file ar.species "
        "(../particle.cpp:711)"),
    "surf_react_probability_above_one": (
        bad_prob[1],
        "ERROR: Surface reaction probability for a species > 1.0 "
        "(../surf_react_prob.cpp:287)"),
}
quoted_ok = True
for name, (txt, msg) in QUOTED.items():
    if msg not in errors(txt):
        quoted_ok = False
        print(f"UNEXPECTED: {name} printed {errors(txt)[:1]} not {msg!r}")

staging_is_the_only_difference = (staged[0] == 0 and unstaged[0] != 0)
probability_is_the_only_difference = (good_prob[0] == 0 and bad_prob[0] != 0)
balance_runs_and_moves_nothing = (
    bal_rc == 0 and len(migrated) == 2
    and all(l.endswith("migrated 0 cells") for l in migrated)
    and len(bal_rows) >= 2)

print(f"every_quoted_setup_message_is_verbatim={quoted_ok}")
print(f"the_same_deck_runs_when_the_species_file_is_staged="
      f"{staging_is_the_only_difference}")
print(f"the_same_deck_runs_when_the_probability_is_in_range="
      f"{probability_is_the_only_difference}")
print(f"the_probability_is_rejected_at_READ_time_before_any_stats_line="
      f"{not stats_rows(bad_prob[1])[1]}")
print(f"balance_grid_on_a_serial_build_migrates_nothing_and_still_runs="
      f"{balance_runs_and_moves_nothing}")
print(f"balance_grid_prints_the_same_line_for_both_styles="
      f"{len(set(migrated)) == 1 if migrated else False}")

ok = (quoted_ok and staging_is_the_only_difference
      and probability_is_the_only_difference
      and not stats_rows(bad_prob[1])[1]
      and balance_runs_and_moves_nothing)
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
