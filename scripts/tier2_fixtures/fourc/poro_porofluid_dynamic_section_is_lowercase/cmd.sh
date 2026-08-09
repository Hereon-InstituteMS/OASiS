#!/bin/bash
# Tier-2 for fourc::porous_media#0 — the porofluid time-integration section is
# spelled `porofluid_dynamic` (lower case, underscore).  Writing it the way
# every other 4C section is written, `POROFLUID DYNAMIC`, is a HARD ABORT at
# parse time, not a silent fallback to defaults.
#
# Two arms on the upstream porofluid deck:
#   good : porofluid_dynamic:   -> runs, 7/7 result tests pass
#   bad  : POROFLUID DYNAMIC:   -> exit 1 before anything is set up
#
# The bad arm's diagnostic is asserted verbatim.  So is the ABSENCE of an
# 'unknown section' banner and of any completed time step: an earlier version of
# this knowledge entry predicted a silent fallback with uniformly-default result
# fields, which would have shown up as result-test failures.  There are none,
# because the run never starts.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream porofluid_pressure_based_2D_quad4.4C.yaml) || exit 3
grep -q '^porofluid_dynamic:' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
# 4C resolves a deck's relative SOLVER xml paths against the DECK's directory,
# so the mutants need the upstream xml/ tree next to them.
ln -s "$(dirname "$BASE")/xml" "$TMP/xml"

cp "$BASE" "$TMP/good.yaml"
sed 's/^porofluid_dynamic:/POROFLUID DYNAMIC:/' "$BASE" > "$TMP/bad.yaml"

probe GOOD "$TMP/good.yaml"
probe BAD  "$TMP/bad.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
grep -m1 -F "Section 'POROFLUID DYNAMIC' is not a valid section name." "$TMP/BAD.log"
grep -m1 -oF "4C_io_input_file.cpp" "$TMP/BAD.log"
# The predicted silent fallback would have produced result-test failures.
echo "BAD_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/BAD.log")"
echo "BAD_TIME_STEPS_STARTED=$(grep -c 'PORO MULTIPHASE FLUID SOLVER' "$TMP/BAD.log")"
# ...and the 'unknown section' wording the entry used to quote does not exist.
echo "CLAIMED_UNKNOWN_SECTION_TEXT=$(grep -ci 'unknown section' "$TMP/BAD.log")"
exit 0
