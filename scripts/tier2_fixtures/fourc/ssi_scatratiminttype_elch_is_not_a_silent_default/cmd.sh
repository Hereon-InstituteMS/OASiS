#!/bin/bash
# Tier-2 for fourc::ssi#0 — and a FALSIFICATION of how it was worded.
#
# Claimed: omitting SCATRATIMINTTYPE: Elch in SSI CONTROL "defaults to plain
#          SCATRA DYNAMIC without electrochemical source terms", so the
#          Butler-Volmer current is silently ZERO and nothing changes.
#
# Observed: nothing silent happens.  Deleting the line from the upstream
# electrode deck makes the two electrochemical unknowns (concentration and
# potential) look like two ordinary transported scalars, and monolithic SSI
# refuses outright:
#
#     Since the ssi_monolithic framework is only implemented for usage in
#     combination with volume change laws 'MAT_InelasticDefgradLinScalarIso' or
#     'MAT_InelasticDefgradLinScalarAniso' so far and these laws are implemented
#     for only one transported scalar at the moment it is not reasonable to use
#     them with more than one transported scalar. ...
#
# from src/ssi/4C_ssi_monolithic.cpp, exit 1 at SsiMono::setup().  So the entry's
# rule (set it for any electrode problem) is right; its Signal was not.  The run
# never reaches a time step, so no BV current is computed at all — asserted.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssi_mono_3D_1hex8_elch_funct_growthlaw.4C.yaml) || exit 3
grep -q '  SCATRATIMINTTYPE: "Elch"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/elch.yaml"
grep -v '  SCATRATIMINTTYPE: "Elch"' "$BASE" > "$TMP/noelch.yaml"

probe ELCH   "$TMP/elch.yaml"
probe NOELCH "$TMP/noelch.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/ELCH.log"
grep -m1 -F "it is not reasonable to use them with more than one transported scalar" "$TMP/NOELCH.log"
grep -m1 -oF "4C_ssi_monolithic.cpp" "$TMP/NOELCH.log"
echo "FAILS_IN_SSIMONO_SETUP=$(grep -c 'SsiMono::setup' "$TMP/NOELCH.log")"
# Nothing runs, so there is no silently-zero current to observe.
echo "NOELCH_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/NOELCH.log")"
echo "NOELCH_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOELCH.log")"
exit 0
