#!/bin/bash
# Tier-2 for fourc::reduced_airways#6 — the element keyword is RED_AIRWAY or
# RED_ACINUS, and neither claimed diagnostic exists.
#
# Claimed: "input parser abort `unknown element type for ELEMENT_BLOCK X` or
#          runtime `elementType BEAM3R has no RedAirway implementation`".
# Observed, on upstream red_airway_one_acinus_NeoHookean:
#   * the spelling the entry itself used, REDAIRWAY, is rejected by the ParObject
#     factory with "Unknown type 'REDAIRWAY' of finite element" from
#     core/comm/src/4C_comm_parobjectfactory.cpp — no ELEMENT_BLOCK, no block id.
#   * BEAM3R does not reach any runtime check at all: it dies in the element
#     reader with "Required 'one_of' not found in input line", because BEAM3R
#     wants beam data the airway line does not carry.
#   * RED_AIRWAY parses as an element but rejects the acinus TYPE, and the
#     message enumerates the airway TYPEs 4C really has.
# Neither claimed string appears anywhere.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream red_airway_one_acinus_NeoHookean.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" base.yaml
grep -q '1 RED_ACINUS LINE2 1 2 MAT 2 TYPE NeoHookean' base.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

sed 's/1 RED_ACINUS LINE2/1 REDAIRWAY LINE2/'  base.yaml > redairway.yaml
sed 's/1 RED_ACINUS LINE2/1 BEAM3R LINE2/'     base.yaml > beam.yaml
sed 's/1 RED_ACINUS LINE2/1 RED_AIRWAY LINE2/' base.yaml > redair.yaml

probe BASE      base.yaml
probe REDAIRWAY redairway.yaml
probe BEAM      beam.yaml
probe REDAIR    redair.yaml

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "Unknown type 'REDAIRWAY' of finite element" "$TMP/REDAIRWAY.log"
grep -m1 -F "4C_comm_parobjectfactory.cpp" "$TMP/REDAIRWAY.log"
grep -m1 -F "Required 'one_of' not found in input line" "$TMP/BEAM.log"
grep -m1 -F "Valid options are: CompliantResistive|ConvectiveViscoElasticRLC|InductoResistive|RLC|Resistive|ViscoElasticRLC" "$TMP/REDAIR.log"
# The two claimed diagnostics do not exist.
echo "CLAIMED_ELEMENT_BLOCK_TEXT=$(cat "$TMP"/*.log | grep -ci 'unknown element type for ELEMENT_BLOCK')"
echo "CLAIMED_BEAM3R_RUNTIME_TEXT=$(cat "$TMP"/*.log | grep -ci 'has no RedAirway implementation')"
exit 0
