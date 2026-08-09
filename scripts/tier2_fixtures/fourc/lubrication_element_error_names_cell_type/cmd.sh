#!/bin/bash
# Tier-2 for fourc::lubrication#1 — LUBRICATION elements really are (d-1)-manifold
# surface elements, but the diagnostic you get for a 3D cell type is NOT the one
# the knowledge entry claimed.
#
# Claimed:  'unsupported element type for lubrication' from 4C_lubrication_factory.cpp
# Observed: "Element 'LUBRICATION' does not seem to know cell type 'hex8'." from
#           core/fem/src/general/element/4C_fem_general_element_definition.cpp
#           line 29, thrown by the generic ElementReader before any lubrication
#           code runs.  Nothing named 4C_lubrication_factory.cpp exists, and the
#           word "unsupported" never appears.
#
# Two arms on the upstream slider-bearing deck: the untouched QUAD4 mesh, and the
# same deck with element 1 rewritten as a HEX8 with eight nodes.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream lubrication_sb_2d.4C.yaml) || exit 3
grep -q '"1 LUBRICATION QUAD4 1 2 3 4 MAT 3"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/quad4.yaml"
sed 's/1 LUBRICATION QUAD4 1 2 3 4 MAT 3/1 LUBRICATION HEX8 1 2 3 4 5 6 7 8 MAT 3/' \
    "$BASE" > "$TMP/hex8.yaml"

probe QUAD4 "$TMP/quad4.yaml"
probe HEX8  "$TMP/hex8.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/QUAD4.log"
grep -m1 -F "Element 'LUBRICATION' does not seem to know cell type 'hex8'." "$TMP/HEX8.log"
grep -m1 -F "4C_fem_general_element_definition.cpp" "$TMP/HEX8.log"
# The failure is in the generic element reader, not in any lubrication factory:
# the claimed file and phrase are absent.
echo "CLAIMED_UNSUPPORTED_ELEMENT_TEXT=$(grep -ci 'unsupported element type' "$TMP/HEX8.log")"
echo "CLAIMED_LUBRICATION_FACTORY_FILE=$(grep -c '4C_lubrication_factory' "$TMP/HEX8.log")"
# ...and the abort happens during mesh reading, before the Reynolds solver exists.
echo "REACHED_TIME_LOOP=$(grep -c '^TIME:' "$TMP/HEX8.log")"
exit 0
