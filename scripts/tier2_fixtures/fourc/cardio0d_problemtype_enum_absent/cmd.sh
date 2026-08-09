#!/bin/bash
# Tier-2 for fourc::cardiovascular0d#4 — there is no Cardiovascular0D problem
# type, and a standalone 0D heart is a Structure problem with a dummy element.
#
# Claim: "Cardiovascular0D is typically used with FLUID or FSI, NOT standalone
#        ... 4C's Cardiovascular0D adapter requires a parent field (PROBLEMTYPE:
#        Fluid or Structure with this condition applied)."
# Observed:
#   * PROBLEMTYPE: "Cardiovascular0D" is not a value 4C knows. The parser prints
#     the whole enum, and neither Cardiovascular0D nor any FSI-with-0D entry is in
#     it. "Structure" is.
#   * standalone IS supported, contrary to the claim, and the upstream deck
#     cardiovascular0d_syspulcirculation_0d_heart is exactly that: a closed-loop
#     sys-pul circulation with VENTRICLE_MODEL "0D", carried by a single dummy
#     SOLID HEX8 whose eight dispx are all result-tested to 0. It exits 0 with all
#     24 result tests correct. The "parent field" is a formality, not a physical
#     coupling.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream cardiovascular0d_syspulcirculation_0d_heart.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" heart.yaml
grep -q '  PROBLEMTYPE: "Structure"' heart.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  VENTRICLE_MODEL: "0D"' heart.yaml    || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '1 SOLID HEX8' heart.yaml               || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

STANDALONE_PROBLEMTYPE=Cardiovascular0D
sed "s/  PROBLEMTYPE: \"Structure\"/  PROBLEMTYPE: \"$STANDALONE_PROBLEMTYPE\"/" heart.yaml > standalone.yaml

probe BASE       heart.yaml
probe STANDALONE standalone.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "BASE_SOLID_ELEMENTS=$(grep -c '^  - "1 SOLID HEX8' heart.yaml)"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "Could not match this input" "$TMP/STANDALONE.log"
grep -m1 -F "Candidate deprecated_selection 'PROBLEMTYPE' has wrong value, possible values:" "$TMP/STANDALONE.log"
N_C0D=$(grep -c 'possible values:.*|Cardiovascular0D|' "$TMP/STANDALONE.log")
N_STRUCT=$(grep -c 'possible values:.*|Structure|' "$TMP/STANDALONE.log")
echo "PROBLEMTYPE_LIST_HAS_CARDIOVASCULAR0D=$N_C0D"
echo "PROBLEMTYPE_LIST_HAS_STRUCTURE=$N_STRUCT"
exit 0
