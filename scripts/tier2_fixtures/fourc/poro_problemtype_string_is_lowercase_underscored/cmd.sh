#!/bin/bash
# Tier-2 for fourc::porous_media#6 — the porofluid problem types are the ONLY
# lower-case entries in 4C's PROBLEMTYPE enum, and they are matched exactly.
# `POROFLUID_PRESSURE_BASED` and `Porofluid` are both rejected.
#
# Claimed:  parser abort `unknown PROBLEMTYPE` listing the legal strings.
# Observed: the listing half is right, the wording is not.  4C says
#
#     Could not match this input ... [!] Candidate deprecated_selection
#     'PROBLEMTYPE' has wrong value, possible values: Ale|...|
#     porofluid_pressure_based|porofluid_pressure_based_elasticity|
#     porofluid_pressure_based_elasticity_scatra
#
# from core/io/src/4C_io_input_spec_builders.cpp.  The phrase 'unknown
# PROBLEMTYPE' never appears; asserted as a zero count.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream porofluid_pressure_based_2D_quad4.4C.yaml) || exit 3
ln -s "$(dirname "$BASE")/xml" "$TMP/xml"
grep -q 'PROBLEMTYPE: "porofluid_pressure_based"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/lower.yaml"
sed 's/PROBLEMTYPE: "porofluid_pressure_based"/PROBLEMTYPE: "POROFLUID_PRESSURE_BASED"/' "$BASE" > "$TMP/upper.yaml"
sed 's/PROBLEMTYPE: "porofluid_pressure_based"/PROBLEMTYPE: "Porofluid"/'                "$BASE" > "$TMP/camel.yaml"

probe LOWER "$TMP/lower.yaml"
probe UPPER "$TMP/upper.yaml"
probe CAMEL "$TMP/camel.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/LOWER.log"
grep -m1 -F "Could not match this input" "$TMP/UPPER.log"
grep -m1 -oF "'PROBLEMTYPE' has wrong value, possible values:" "$TMP/UPPER.log"
grep -m1 -oF "4C_io_input_spec_builders.cpp" "$TMP/UPPER.log"
# The legal lower-case strings ARE printed, which is the useful half of the claim.
echo "UPPER_LISTS_LEGAL_LOWERCASE=$(grep -c 'porofluid_pressure_based_elasticity_scatra' "$TMP/UPPER.log")"
echo "CAMEL_LISTS_LEGAL_LOWERCASE=$(grep -c 'porofluid_pressure_based_elasticity_scatra' "$TMP/CAMEL.log")"
# The quoted wording does not exist in either rejection.
echo "CLAIMED_UNKNOWN_PROBLEMTYPE_TEXT=$(grep -ci 'unknown PROBLEMTYPE' "$TMP/UPPER.log" "$TMP/CAMEL.log" | awk -F: '{s+=$2} END {print s+0}')"
exit 0
