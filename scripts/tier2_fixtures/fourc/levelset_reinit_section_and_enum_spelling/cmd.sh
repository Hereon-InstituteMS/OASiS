#!/bin/bash
# Tier-2 for fourc::level_set#1 -- reinitialisation is real, but both spellings
# the entry gave are wrong.
#
# Claimed: "Set LEVELSET CONTROL: REINITIALIZATION: signed_distance_function".
# Real:    the section is LEVEL-SET CONTROL/REINITIALIZATION (hyphen, and a
#          sub-section), and the value is Signed_Distance_Function.  The enum is
#          case-sensitive: the lower-case spelling is rejected and 4C prints the
#          four admissible values.  The properly spelled value works.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream levelset_elliptic_reinit_lin.4C.yaml) || exit 3
grep -q "^LEVEL-SET CONTROL/REINITIALIZATION:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  REINITIALIZATION: "EllipticEq"' "$BASE"     || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/base.yaml"
sed 's|^LEVEL-SET CONTROL/REINITIALIZATION:|LEVELSET CONTROL:\n  REINITIALIZATION: "signed_distance_function"\nLEVEL-SET CONTROL/REINITIALIZATION:|' "$BASE" > "$TMP/badsection.yaml"
sed 's/  REINITIALIZATION: "EllipticEq"/  REINITIALIZATION: "signed_distance_function"/' "$BASE" > "$TMP/lowercase.yaml"
sed 's/  REINITIALIZATION: "EllipticEq"/  REINITIALIZATION: "Signed_Distance_Function"/' "$BASE" > "$TMP/properspelling.yaml"

probe BASE           "$TMP/base.yaml"
probe BADSECTION     "$TMP/badsection.yaml"
probe LOWERCASE      "$TMP/lowercase.yaml"
probe PROPERSPELLING "$TMP/properspelling.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BASE.log")"
# the claimed section name
grep -m1 -F "Section 'LEVELSET CONTROL' is not a valid section name." "$TMP/BADSECTION.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/BADSECTION.log"
# the claimed enum spelling
grep -m1 -F "possible values: EllipticEq|None|Signed_Distance_Function|Sussman" "$TMP/LOWERCASE.log"
# the correct spelling runs and still satisfies the deck
grep -m1 -F "processor 0 finished normally" "$TMP/PROPERSPELLING.log"
echo "PROPERSPELLING_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/PROPERSPELLING.log")"
exit 0
