#!/bin/bash
# Tier-2 for fourc::multiscale#6 — a FALSIFICATION of the remedy it prescribes.
#
# Claimed: "macro NOX residual halves slowly (linear-rate, not quadratic Newton)
#          and MAXITER is hit on every macro step; switching MICRO_TANGENT to
#          ALGORITHMIC restores quadratic convergence."
# Observed: there is no MICRO_TANGENT key in 4C. Zero occurrences in the binary's
#          own --parameters schema. Writing it into MAT_Struct_Multiscale — the
#          only place it could plausibly live — is rejected outright with
#          'Could not match this input' and 'The following data remains unused'.
#          MAT_Struct_Multiscale takes MICROFILE, MICRODIS_NUM and
#          RUNTIMEOUTPUT_GP; there is no tangent switch on it at all.
#
# The advice was therefore unfollowable: an agent that acted on it would get a
# rejected input file, not a faster Newton.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sohex8_multiscale_macro.4C.yaml) || exit 3
MICRO=$(upstream sohex8_multiscale_micro.mat.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" macro.yaml
cp "$MICRO" .
grep -q "      MICRODIS_NUM: 1" macro.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - macro.yaml tangent.yaml <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "      MICRODIS_NUM: 1"
assert anchor in t
open(sys.argv[2], "w").write(
    t.replace(anchor, anchor + "\n      MICRO_TANGENT: ALGORITHMIC", 1))
PY

probe GOOD    macro.yaml
probe TANGENT tangent.yaml

echo "GOOD_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/GOOD.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
grep -m1 -F "Could not match this input" "$TMP/TANGENT.log"
grep -m1 -F "The following data remains unused" "$TMP/TANGENT.log"
grep -m1 -F "4C_global_data_read.cpp" "$TMP/TANGENT.log"
"$BIN" --parameters 2>/dev/null > params.json
echo "MICRO_TANGENT_IN_SCHEMA=$(grep -c 'MICRO_TANGENT' params.json)"
echo "RUNTIMEOUTPUT_GP_IN_SCHEMA=$(grep -c 'RUNTIMEOUTPUT_GP' params.json)"
exit 0
