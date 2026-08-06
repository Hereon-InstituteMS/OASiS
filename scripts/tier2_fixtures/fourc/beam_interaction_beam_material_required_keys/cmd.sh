#!/bin/bash
# Tier-2 for fourc::beam_interaction#5 — MAT_BeamReissnerElastHyper really does
# require CROSSAREA and MOMINPOL, but 4C says so in a shape nobody would guess.
#
# Claimed: the parser aborts with `MAT_BeamReissnerElastHyper requires CROSSAREA`
#          / `MOMINPOL not specified`. Neither string exists in 4C.
# Observed: 4C reports "Failed to match specification in section 'MATERIALS'",
#          then prints a MATCH TREE in which it tries EVERY material spec in turn
#          and marks the one that nearly fit. The single decisive line is
#              [X] Expected parameter 'CROSSAREA'
#          buried some 600 lines into that tree, after a long list of unrelated
#          candidates like MAT_LinElast1DGrowth that "did not match".
#
# That is the string to grep for when a material card is rejected: `[X] Expected
# parameter '<KEY>'`, not any sentence naming the material.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_static_test1.4C.yaml) || exit 3
grep -q "      CROSSAREA: 1$" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "      MOMINPOL: 0.1406$" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/full.yaml"
grep -v "      CROSSAREA: 1$"    "$BASE" > "$TMP/nocross.yaml"
grep -v "      MOMINPOL: 0.1406$" "$BASE" > "$TMP/nomom.yaml"

probe FULL    "$TMP/full.yaml"
probe NOCROSS "$TMP/nocross.yaml"
probe NOMOM   "$TMP/nomom.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/FULL.log"
grep -m1 -F "Failed to match specification in section 'MATERIALS'" "$TMP/NOCROSS.log"
grep -m1 -F "[X] Expected parameter 'CROSSAREA'" "$TMP/NOCROSS.log"
grep -m1 -F "[X] Expected parameter 'MOMINPOL'" "$TMP/NOMOM.log"
# The decisive line is buried: report how far in, so nobody assumes it is near the top.
echo "CROSSAREA_LINE_BEYOND_100=$([ "$(grep -n -m1 -F "[X] Expected parameter 'CROSSAREA'" "$TMP/NOCROSS.log" | cut -d: -f1)" -gt 100 ] && echo yes || echo no)"
# Unrelated material specs are offered as candidates in the same tree.
echo "TREE_OFFERS_UNRELATED_MATERIALS=$(grep -c "Expected group 'MAT_LinElast1D'" "$TMP/NOCROSS.log")"
# Neither claimed sentence exists.
echo "CLAIMED_REQUIRES_CROSSAREA_TEXT=$(grep -ci 'requires CROSSAREA' "$TMP/NOCROSS.log")"
echo "CLAIMED_MOMINPOL_NOT_SPECIFIED_TEXT=$(grep -ci 'MOMINPOL not specified' "$TMP/NOMOM.log")"
exit 0
