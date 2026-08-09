#!/bin/bash
# Tier-2 for fourc::low_mach#3 -- a missing consistent initial field is not a
# convergence nuisance.  It is a division by zero.
#
# Claimed: "first-step residual norm is very large ... Newton may converge slowly
#          or diverge".
# Observed: nothing converges slowly, because nothing runs.  Switching both
#          INITIALFIELD entries of the upstream heated-channel deck from
#          field_by_function to zero_field starts the temperature at T = 0, and
#          the Sutherland material computes density as p/(R*T).  4C dies with
#          SIGFPE -- "Signal: Floating point exception (8)" / "Signal code:
#          Floating point divide-by-zero (3)" -- with no 4C-level message at all.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }
BASE=$(upstream loma_2d_heated_chan_30x100.4C.yaml) || exit 3
link_xml "$BASE"

grep -q '  INITIALFIELD: "field_by_function"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "    MAT_sutherland:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/consistent.yaml"
sed 's/  INITIALFIELD: "field_by_function"/  INITIALFIELD: "zero_field"/' "$BASE" > "$TMP/zerofield.yaml"

probe CONSISTENT "$TMP/consistent.yaml"
probe ZEROFIELD  "$TMP/zerofield.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/CONSISTENT.log"
echo "CONSISTENT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/CONSISTENT.log")"
echo "ZEROFIELD_INITIALFIELDS_SWITCHED=$(grep -c 'INITIALFIELD: \"zero_field\"' "$TMP/zerofield.yaml")"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/ZEROFIELD.log"
grep -m1 -F "Floating point divide-by-zero (3)" "$TMP/ZEROFIELD.log"
# No 4C diagnostic, no Newton table, no result test.
echo "ZEROFIELD_FOURC_THROW=$(grep -c 'PROC 0 ERROR in' "$TMP/ZEROFIELD.log")"
echo "ZEROFIELD_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/ZEROFIELD.log")"
exit 0
