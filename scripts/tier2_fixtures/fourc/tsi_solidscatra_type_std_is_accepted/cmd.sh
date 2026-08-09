#!/bin/bash
# Tier-2 for fourc::tsi#6 — the SOLIDSCATRA TYPE keyword, executed.  Upstream
# tsi_lincompression_1waydisp, four arms differing only in the TYPE token of the
# two element lines.
#
#   TYPE Undefined  -> runs, every result test CORRECT, exit 0   (the recommendation)
#   TYPE Std        -> runs, every result test CORRECT, exit 0   <- claim says this throws
#   TYPE omitted    -> "Required value 'TYPE' not found in input line"
#                      4C_io_input_spec_builders.cpp — a PARSE error while reading
#                      the element line, not a runtime throw at problem setup
#   TYPE Bogus      -> "The input type Bogus is not valid for SOLIDSCATRA elements!"
#                      4C_solid_scatra_3D_ele_lib.cpp, still during element
#                      reading: no fill_complete, no 'Total wall time for INPUT',
#                      no time step ever starts
#
# So 'TYPE Undefined' is a recommendation, not a requirement: 'TYPE Std' is one
# of the values the enum accepts and it runs to the same answer.  What is fatal
# is a value OUTSIDE the enum, and — separately — omitting the key altogether,
# which the YAML-level element spec catches because TYPE is declared required.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_lincompression_1waydisp.4C.yaml) || exit 3
grep -q "KINEM linear TYPE Undefined" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/undefined.yaml"
sed 's/KINEM linear TYPE Undefined/KINEM linear TYPE Std/'   "$BASE" > "$TMP/std.yaml"
sed 's/KINEM linear TYPE Undefined/KINEM linear/'            "$BASE" > "$TMP/omitted.yaml"
sed 's/KINEM linear TYPE Undefined/KINEM linear TYPE Bogus/' "$BASE" > "$TMP/bogus.yaml"

probe TYPE_UNDEFINED "$TMP/undefined.yaml"
probe TYPE_STD       "$TMP/std.yaml"
probe TYPE_OMITTED   "$TMP/omitted.yaml"
probe TYPE_BOGUS     "$TMP/bogus.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/TYPE_UNDEFINED.log"
# 'TYPE Std' is accepted and gives the same answer as 'TYPE Undefined'.
grep -m1 -F "processor 0 finished normally" "$TMP/TYPE_STD.log"
echo "STD_RESULT_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/TYPE_STD.log")"
echo "UNDEFINED_RESULT_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/TYPE_UNDEFINED.log")"
echo "STD_THROWS=$(grep -c 'not valid for SOLIDSCATRA elements' "$TMP/TYPE_STD.log")"
# Omitting the key is caught by the element spec, not by a runtime check.
grep -m1 -F "Required value 'TYPE' not found in input line" "$TMP/TYPE_OMITTED.log"
grep -m1 -oF "4C_io_input_spec_builders.cpp" "$TMP/TYPE_OMITTED.log"
echo "OMITTED_THROWS_SOLIDSCATRA_MESSAGE=$(grep -c 'not valid for SOLIDSCATRA elements' "$TMP/TYPE_OMITTED.log")"
# A value outside the enum is what really throws — during element reading.
grep -m1 -F "The input type Bogus is not valid for SOLIDSCATRA elements!" "$TMP/TYPE_BOGUS.log"
grep -m1 -oF "4C_solid_scatra_3D_ele_lib.cpp" "$TMP/TYPE_BOGUS.log"
echo "BOGUS_REACHED_FILL_COMPLETE=$(grep -c 'fill_complete' "$TMP/TYPE_BOGUS.log")"
echo "BOGUS_REACHED_FIRST_TIME_STEP=$(grep -c '^TIME: ' "$TMP/TYPE_BOGUS.log")"
exit 0
