#!/bin/bash
# Tier-2 for fourc::beam_interaction#2 — a FALSIFICATION.
#
# Claimed: "too-small SEARCH_RADIUS misses contact pairs — beams pass through each
#          other; too-large radius wastes compute... Typical SEARCH_RADIUS ~ 2-3 *
#          max beam diameter."
# Observed: the beam-to-beam contact framework has NO SEARCH_RADIUS parameter.
#          The only SEARCH_RADIUS in 4C belongs to FLUID BEAM INTERACTION —
#          declared once in src/fbi/src/4C_fbi_input.cpp — which is a different
#          module. Adding SEARCH_RADIUS to BEAM INTERACTION/BEAM TO BEAM CONTACT
#          is rejected outright, and adding it to BINNING STRATEGY is rejected
#          too. What governs the search on a beam-contact run is BINNING
#          STRATEGY's DOMAINBOUNDINGBOX / BIN_SIZE_LOWER_BOUND plus the
#          segmentation angles.
. "$(dirname "$0")/../_lib/preamble.sh"

DECK=beam3eb_static_contact_penalty_linpen_limitdispperiter_twobeamstwisting
BASE=$(upstream "$DECK.4C.yaml") || exit 3
XML=$(upstream "$DECK.xml")      || exit 3
cd "$TMP" || exit 3
cp "$XML" .
cp "$BASE" base.yaml

python3 - base.yaml btb.yaml bin.yaml <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "BEAM INTERACTION/BEAM TO BEAM CONTACT:\n"
assert anchor in t, "upstream deck no longer has the beam-to-beam contact section"
open(sys.argv[2], "w").write(t.replace(anchor, anchor + "  SEARCH_RADIUS: 0.05\n", 1))
anchor2 = "BINNING STRATEGY:\n"
assert anchor2 in t
open(sys.argv[3], "w").write(t.replace(anchor2, anchor2 + "  SEARCH_RADIUS: 0.05\n", 1))
PY

probe BASE base.yaml
probe BTB  btb.yaml
probe BIN  bin.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
grep -m1 -F "Could not match this input" "$TMP/BTB.log"
grep -m1 -F "The following data remains unused" "$TMP/BTB.log"
grep -m1 -F "Could not match this input" "$TMP/BIN.log"
# The one place SEARCH_RADIUS is legal is a different module's section.
"$BIN" --parameters 2>/dev/null > params.json
echo "SEARCH_RADIUS_IN_SCHEMA=$(grep -c 'SEARCH_RADIUS' params.json)"
echo "FBI_SECTION_IN_SCHEMA=$(grep -c 'FLUID BEAM INTERACTION' params.json)"
exit 0
