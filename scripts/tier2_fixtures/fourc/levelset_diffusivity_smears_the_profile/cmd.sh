#!/bin/bash
# Tier-2 for fourc::level_set#0 -- DIFFUSIVITY in a level-set transport really is
# physics, not a numerical dial, and its effect is measurable in one deck.
#
# The upstream Gaussian-hill level-set deck deliberately ships a NON-zero
# MAT_scatra DIFFUSIVITY and pins four values that include that smearing.  Set
# DIFFUSIVITY to 0 and every pinned value moves in the same direction -- the hill
# stays taller, because nothing has diffused the peak away.  Nothing warns: the
# run is healthy, reaches its result test, and simply reports different numbers.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves IFPACK_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream levelset_gaussian_hill_pbc.4C.yaml) || exit 3
grep -q "      DIFFUSIVITY: 0.0006666666666" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/withdiff.yaml"
sed 's/      DIFFUSIVITY: 0.0006666666666/      DIFFUSIVITY: 0.0/' "$BASE" > "$TMP/nodiff.yaml"

probe WITHDIFF "$TMP/withdiff.yaml"
probe NODIFF   "$TMP/nodiff.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHDIFF.log"
echo "WITHDIFF_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHDIFF.log")"
echo "NODIFF_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NODIFF.log")"
echo "NODIFF_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NODIFF.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NODIFF.log"
# The pure-advection field is everywhere ABOVE the diffused one at the pinned nodes.
# At every pinned node the pure-advection value is ABOVE the diffused reference.
echo "NODIFF_PEAK_HIGHER=$(sed -n 's/.*actresult=[[:space:]]*\([-0-9.eE+]*\)[[:space:]]*,[[:space:]]*givenresult=[[:space:]]*\([-0-9.eE+]*\).*/\1 \2/p' "$TMP/NODIFF.log" | awk '$1 > $2 {n++} END {print n+0}')"
echo "DIFFUSIVITY_WARNINGS=$(grep -ciE 'diffusiv.*(ignor|warn)|smear' "$TMP/NODIFF.log")"
exit 0
