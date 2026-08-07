#!/bin/bash
# Tier-2 for fourc::membrane#3 — and a FALSIFICATION of its premise.
#
# The entry warned that "omitting THICK uses default 1.0".  There is no default:
# THICK is required:true on MEMBRANE4 and omitting it is a parse error naming
# the key.  The rest of the entry stands in direction but not in form — the
# response is not a clean linear scaling, because the upstream deck this runs on
# is geometrically nonlinear.
#
# Upstream deck membrane_cyl_new_struc, which runs as shipped and result-tests
# node 21 to 1e-08.
#
#   BASE      THICK 0.1 as shipped -> exit 0
#   THICK_0p2 THICK doubled        -> result test fails, prints the new value
#   NO_THICK  key removed          -> "Required value 'THICK' not found"
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream membrane_cyl_new_struc.4C.yaml) || exit 3
grep -q "THICK 0.1" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/base.4C.yaml"
sed 's/THICK 0.1/THICK 0.2/'  "$BASE" > "$TMP/t02.4C.yaml"
sed 's/ THICK 0.1//'          "$BASE" > "$TMP/nothick.4C.yaml"

probe BASE      "$TMP/base.4C.yaml"
probe THICK_0p2 "$TMP/t02.4C.yaml"
probe NO_THICK  "$TMP/nothick.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -oE "is WRONG --> actresult=[^,]*" "$TMP/THICK_0p2.log"
grep -m1 -F "Required value 'THICK' not found in input line" "$TMP/NO_THICK.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/NO_THICK.log"
echo "NO_THICK_STEPS=$(grep -c '^Finalised step' "$TMP/NO_THICK.log")"
exit 0
