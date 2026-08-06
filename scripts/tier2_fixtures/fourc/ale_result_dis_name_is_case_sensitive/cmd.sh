#!/bin/bash
# Tier-2 for fourc::ale#5 — RESULT DESCRIPTION's DIS name is case-sensitive, and
# the failure it produces is NOT the one the entry originally claimed.
#
# Claimed:  writing DIS: "ALE" raises 'discretisation ALE not found'.
# Observed: nothing of the sort.  4C runs the whole simulation, reaches the
#           result-test manager, silently matches ZERO of the tests, and aborts
#           with "expected 2 tests but performed 0" from
#           core/utils/src/result_test/4C_utils_result_test.cpp.
#
# The distinction matters for an agent reading the log: the run LOOKS successful
# up to and including "Restart written in step 2", the mis-spelled field name is
# never echoed, and the only clue is a count mismatch at the very end.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ale2d_solid.4C.yaml) || exit 3
grep -q 'DIS: "ale"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/lower.yaml"
sed 's/DIS: "ale"/DIS: "ALE"/' "$BASE" > "$TMP/upper.yaml"

probe LOWER "$TMP/lower.yaml"
probe UPPER "$TMP/upper.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/LOWER.log"
grep -m1 -F "expected 2 tests but performed 0" "$TMP/UPPER.log"
grep -m1 -F "4C_utils_result_test.cpp" "$TMP/UPPER.log"
# The claimed diagnostic does not exist anywhere in the output.
echo "CLAIMED_DISCRETISATION_NOT_FOUND_TEXT=$(grep -ci 'discretisation ALE not found' "$TMP/UPPER.log")"
# ...and the mis-spelled name is never echoed back to the reader.
echo "MISSPELLED_NAME_ECHOED=$(grep -c 'DIS: "ALE"' "$TMP/UPPER.log")"
exit 0
