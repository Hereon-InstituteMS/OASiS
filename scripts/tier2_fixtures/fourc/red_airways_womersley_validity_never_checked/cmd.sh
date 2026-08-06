#!/bin/bash
# Tier-2 for fourc::reduced_airways#4 — the long-wavelength / Womersley
# assumption is never checked, computed, or mentioned by 4C.
#
# Claim: "Reduced model assumes LONG WAVELENGTH (Womersley number restrictions,
#        Wo < O(1)) ... Check Wo = R * sqrt(omega / nu) before trusting
#        reduced-model results."
# Observed, on upstream red_airway_3airway_2acinus_awacinter: the deck drives the
# inlet at 0.01 Hz (period 100). Re-tuning that single FUNCT1 to 10 Hz - HFOV, a
# thousand-fold jump in Womersley number for the same geometry and the same
# viscosity - changes nothing about how 4C behaves. The run completes its 5000
# steps and exits through the result test, and the string "omersley" appears
# exactly 0 times in either log. There is no validity warning of any kind; the
# only difference is the answer, node 2 pressure 29.868 -> 16.040.
# The user is on their own for Wo: 4C will not compute it.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream red_airway_3airway_2acinus_awacinter.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" base.yaml
grep -qF 'SYMBOLIC_FUNCTION_OF_TIME: "15*(sin(pi*t/50-pi/2)+1)"' base.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

sed 's|15\*(sin(pi\*t/50-pi/2)+1)|15*(sin(2*pi*10*t-pi/2)+1)|' base.yaml > hfov.yaml
cmp -s base.yaml hfov.yaml && { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

probe BASE base.yaml
probe HFOV hfov.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "HFOV_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/HFOV.log")"
echo "HFOV_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/HFOV.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "pressure at node   2" "$TMP/HFOV.log"
# 4C never names the assumption it is running outside of.
echo "WOMERSLEY_MENTIONS=$(grep -ci 'omersley' "$TMP/BASE.log" "$TMP/HFOV.log" | awk -F: '{s+=$2} END {print s+0}')"
echo "HFOV_VALIDITY_WARNINGS=$(grep -ciE 'long wavelength|assumption|out of range|not valid|reduced model' "$TMP/HFOV.log")"
exit 0
