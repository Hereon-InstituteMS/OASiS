#!/bin/bash
# Tier-2 for fourc::ssi#3 — and a FALSIFICATION of how it was worded.
#
# Claimed: "default VELOCITYFIELD: zero makes the scatra ignore the structural
#          motion; transport in a deforming domain is wrong by O(|v_s|) —
#          visible as a stationary concentration field even with large
#          structural deformation."
#
# Observed: 4C will not run it.  Setting VELOCITYFIELD: "zero" in
# SCALAR TRANSPORT DYNAMIC on the upstream monolithic SSI deck gives
#
#     Invalid type of velocity field for scalar-structure interaction!
#     src/ssi/4C_ssi_monolithic.cpp
#
# from SsiMono::init(), exit 1.  The check happens at init, before any field is
# built, so there is no stationary-concentration artefact to inspect — asserted
# as a zero result-test count.  The rule stands, the failure mode does not.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssi_mono_3D_1hex8_scatra.4C.yaml) || exit 3
grep -q '  VELOCITYFIELD: "Navier_Stokes"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/ns.yaml"
sed 's/  VELOCITYFIELD: "Navier_Stokes"/  VELOCITYFIELD: "zero"/' "$BASE" > "$TMP/zero.yaml"

probe NS   "$TMP/ns.yaml"
probe ZERO "$TMP/zero.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/NS.log"
grep -m1 -F "Invalid type of velocity field for scalar-structure interaction!" "$TMP/ZERO.log"
grep -m1 -oF "4C_ssi_monolithic.cpp" "$TMP/ZERO.log"
# Rejected at init, before the fields exist.
echo "FAILS_IN_SSIMONO_INIT=$(grep -c 'SsiMono::init' "$TMP/ZERO.log")"
echo "ZERO_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/ZERO.log")"
echo "ZERO_TIME_STEPS_STARTED=$(grep -c 'Checking results of' "$TMP/ZERO.log")"
exit 0
