#!/bin/bash
# Tier-2 for fourc::ssi#4 — and a FALSIFICATION of how it was worded.
#
# Claimed: CONVFORM: convective on a moving-mesh scatra "produces O(dv_s/dt)
#          mass-balance drift over time — integral of concentration drifts by
#          1-5% per cycle", i.e. a slow quiet error you have to go looking for.
#
# Observed: 4C refuses to start.  Switching CONVFORM from "conservative" to
# "convective" on the upstream monolithic SSI deck gives, from
# src/ssi/4C_ssi_base.cpp at SSIBase::init(), exit 1:
#
#     Inconsistent scalar transport formulation on a deforming domain: The
#     scalar is defined as volume-referenced (IS_INTENSIVE_SCALAR = false),
#     therefore the conservative form is required to account for volume changes.
#     Please set 'CONVFORM' to 'conservative' in the SCALAR TRANSPORT DYNAMIC
#     section.
#
# That message also names the escape hatch the entry never mentions: the
# requirement is tied to IS_INTENSIVE_SCALAR, so it is the scalar's definition,
# not SSI as such, that forces the conservative form.  No drift is ever
# integrated, because no time step runs — asserted.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssi_mono_3D_1hex8_scatra.4C.yaml) || exit 3
grep -q '  CONVFORM: "conservative"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/cons.yaml"
sed 's/  CONVFORM: "conservative"/  CONVFORM: "convective"/' "$BASE" > "$TMP/conv.yaml"

probe CONS "$TMP/cons.yaml"
probe CONV "$TMP/conv.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/CONS.log"
grep -m1 -F "Inconsistent scalar transport formulation on a deforming domain" "$TMP/CONV.log"
grep -m1 -F "Please set 'CONVFORM' to 'conservative' in the SCALAR TRANSPORT DYNAMIC section." "$TMP/CONV.log"
grep -m1 -oF "4C_ssi_base.cpp" "$TMP/CONV.log"
# The condition is stated in terms of IS_INTENSIVE_SCALAR, which the entry omitted.
echo "DIAGNOSTIC_NAMES_IS_INTENSIVE_SCALAR=$(grep -c 'IS_INTENSIVE_SCALAR' "$TMP/CONV.log")"
# Rejected at init: no drift can accumulate because no step is taken.
echo "CONV_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/CONV.log")"
echo "CONV_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/CONV.log")"
exit 0
