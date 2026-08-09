#!/bin/bash
# Tier-2 for fourc::level_set#3 -- turning SUPG off on an advection-dominated
# level-set transport does change the answer, but nothing like the amount the
# entry claimed, and it produces no visible ringing to spot.
#
# Claimed: "STABTYPE: no_stabilization on a high-Pe level-set transport produces
#          ringing across element edges (5-10% amplitude that does not damp)".
# Observed: on the upstream Gaussian-hill deck -- whose default is SUPG with
#          DEFINITION_TAU: Taylor_Hughes_Zarins, exactly the recommended setting
#          -- STABTYPE: no_stabilization fails all four pinned values, but every
#          deviation is well under 1% of the value, not 5-10%.  The failure is
#          only detectable against a reference, which is the real lesson.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves IFPACK_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }
pairs() { sed -n 's/.*actresult=[[:space:]]*\([-0-9.eE+]*\)[[:space:]]*,[[:space:]]*givenresult=[[:space:]]*\([-0-9.eE+]*\).*/\1 \2/p' "$1"; }

BASE=$(upstream levelset_gaussian_hill_pbc.4C.yaml) || exit 3
grep -q "^SCALAR TRANSPORT DYNAMIC:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "STABTYPE" "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/supg.yaml"
sed 's|^SCALAR TRANSPORT DYNAMIC:|SCALAR TRANSPORT DYNAMIC/STABILIZATION:\n  STABTYPE: "no_stabilization"\nSCALAR TRANSPORT DYNAMIC:|' "$BASE" > "$TMP/nostab.yaml"

probe SUPG   "$TMP/supg.yaml"
probe NOSTAB "$TMP/nostab.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SUPG.log"
echo "SUPG_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SUPG.log")"
echo "NOSTAB_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOSTAB.log")"
echo "NOSTAB_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOSTAB.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NOSTAB.log"
# How big is the claimed 5-10% ringing, really?
echo "NOSTAB_DEVIATIONS_ABOVE_5PCT=$(pairs "$TMP/NOSTAB.log" | awk '{d=($1-$2)/$2; if (d<0) d=-d; if (d>0.05) n++} END {print n+0}')"
echo "NOSTAB_DEVIATIONS_ABOVE_1PCT=$(pairs "$TMP/NOSTAB.log" | awk '{d=($1-$2)/$2; if (d<0) d=-d; if (d>0.01) n++} END {print n+0}')"
echo "NOSTAB_NAN=$(grep -ci 'nan' "$TMP/NOSTAB.log")"
exit 0
