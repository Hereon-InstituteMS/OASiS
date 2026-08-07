#!/bin/bash
# Tier-2 for fourc::beam_interaction#7 — there is no PROBLEMTYPE
# 'BeamInteraction'.
#
# The rule holds; the Signal did not. 4C never says 'unknown problem type'; it
# says "Could not match this input" and prints the whole allowed enum. What the
# enum DOES contain is worth pinning, because it is what an agent should reach
# for instead: Polymer_Network (the 19 upstream beam-network decks) and
# Fluid_Beam_Interaction, plus plain Structure for beam-to-beam contact driven by
# the BEAM INTERACTION section.
. "$(dirname "$0")/../_lib/preamble.sh"

cat > "$TMP/bad.yaml" <<'YAML'
PROBLEM TYPE:
  PROBLEMTYPE: "BeamInteraction"
YAML

probe BAD "$TMP/bad.yaml"

grep -m1 -F "Could not match this input" "$TMP/BAD.log"
grep -m1 -oF "has wrong value, possible values:" "$TMP/BAD.log"
echo "CLAIMED_UNKNOWN_PROBLEM_TYPE_TEXT=$(grep -ci 'unknown problem type' "$TMP/BAD.log")"

ENUM=$(grep -m1 -o "possible values: [A-Za-z_|0-9]*" "$TMP/BAD.log")
echo "ENUM_HAS_BEAMINTERACTION=$(printf '%s|' "$ENUM" | grep -c '|BeamInteraction|')"
echo "ENUM_HAS_POLYMER_NETWORK=$(printf '%s|' "$ENUM" | grep -c '|Polymer_Network|')"
echo "ENUM_HAS_FLUID_BEAM_INTERACTION=$(printf '%s|' "$ENUM" | grep -c '|Fluid_Beam_Interaction|')"

# The BEAM INTERACTION section itself is real and lives under a Structure run.
"$BIN" --parameters 2>/dev/null > "$TMP/params.json"
echo "BEAM_INTERACTION_SECTION_IN_SCHEMA=$(grep -c 'BEAM INTERACTION' "$TMP/params.json")"
exit 0
