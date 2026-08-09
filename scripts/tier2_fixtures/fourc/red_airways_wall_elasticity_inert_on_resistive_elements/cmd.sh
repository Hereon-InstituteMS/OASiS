#!/bin/bash
# Tier-2 for fourc::reduced_airways#3 — airway wall compliance is an ELEMENT-line
# option, it is per airway TYPE, and MAT_redAirway does not exist.
#
# Claimed: "a rigid-airway model under high resistive load fails to show the
#          expected collapse instability ... Use COMPLIANCE in MAT_redAirway".
# Observed, on upstream red_airway_1airway_acinus_collapsible (TYPE Resistive with
# the Bates-Irvin collapse model, AirwayColl 1):
#   * there is no MAT_redAirway in 4C. Renaming the acinus material to it is a
#     parse failure in section 'MATERIALS'; the airway itself uses MAT_fluid.
#   * WallElasticity 0.0 -> 5000.0 on a Resistive airway changes NOTHING: both
#     result tests stay CORRECT with abs(diff) exactly 0.0. The parser accepts the
#     value and the element never reads it.
#   * the switch that does control collapse/reopening is AirwayColl on the element
#     line. Turning it off (1 -> 0) moves node 2 pressure 1.1964 -> 0.0899, a
#     factor 13, and the acinar volume 1.1458 -> 1.0108.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream red_airway_1airway_acinus_collapsible.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" base.yaml
grep -q 'AirwayColl 1 ' base.yaml     || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'WallElasticity 0.0 ' base.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'MAT_0D_MAXWELL_ACINUS_EXPONENTIAL:' base.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

sed 's/WallElasticity 0.0 /WallElasticity 5000.0 /'  base.yaml > wallstiff.yaml
sed 's/AirwayColl 1 /AirwayColl 0 /'                 base.yaml > rigid.yaml
sed 's/MAT_0D_MAXWELL_ACINUS_EXPONENTIAL:/MAT_redAirway:/' base.yaml > matred.yaml

probe BASE      base.yaml
probe WALLSTIFF wallstiff.yaml
probe RIGID     rigid.yaml
probe MATRED    matred.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "WALLSTIFF_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/WALLSTIFF.log")"
echo "WALLSTIFF_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/WALLSTIFF.log")"
echo "RIGID_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/RIGID.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/WALLSTIFF.log"
grep -m1 -F "pressure at node   2" "$TMP/WALLSTIFF.log"
grep -m1 -F "pressure at node   2" "$TMP/RIGID.log"
grep -m1 -F "acini_volume at element   2" "$TMP/RIGID.log"
# a 5000-fold WallElasticity change on a Resistive airway is bit-for-bit inert
if grep -q 'pressure at node   2.*is CORRECT, abs(diff)= 0.00000000000000000e+00' "$TMP/WALLSTIFF.log"; then
  echo "VERDICT: WALLELASTICITY_MOVES_A_RESISTIVE_AIRWAY=no"
else
  echo "VERDICT: WALLELASTICITY_MOVES_A_RESISTIVE_AIRWAY=yes"
fi
echo "WALLSTIFF_INERT_PARAMETER_WARNINGS=$(grep -ciE 'WallElasticity|unused|ignored' "$TMP/WALLSTIFF.log")"
grep -m1 -F "Failed to match specification in section 'MATERIALS'." "$TMP/MATRED.log"
echo "MAT_REDAIRWAY_EXISTS=$(grep -c 'Expected group .MAT_redAirway' "$TMP/MATRED.log")"
exit 0
