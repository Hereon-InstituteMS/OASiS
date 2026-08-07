#!/bin/bash
# Tier-2 for fourc::structural_mechanics#8 — a beam mesh needs a BEAM3* element
# type, and the diagnostic for using SOLID or WALL names the CELL type, not the
# element type.
#
# Upstream deck beam3r_line2_static_test1.4C.yaml (five BEAM3R LINE2 cells with
# TRIADS), one token changed:
#
#   BEAM3R LINE2 -> runs, exit 0
#   SOLID  LINE2 -> Element 'SOLID' does not seem to know cell type 'line2'.
#   WALL   LINE2 -> Element 'WALL'  does not seem to know cell type 'line2'.
#
# Both throws come from the generic element-definition lookup in
# core/fem/general/element/4C_fem_general_element_definition.cpp, because SOLID
# and WALL are perfectly well registered — they just register volume and surface
# cell types only.  An earlier version of the entry attributed this to 'Unknown
# type' from parobjectfactory.cpp; that is a real 4C message but a different
# code path, and it is never reached here.  CLAIMED_PAROBJECTFACTORY_TEXTS=0
# keeps the correction pinned.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_static_test1.4C.yaml) || exit 3
for needle in 'BEAM3R LINE2' 'TRIADS' 'MAT_BeamReissnerElastHyper:'; do
  grep -qF "$needle" "$BASE" || {
    echo "FIXTURE_ABORT=upstream_deck_changed (missing: $needle)"; exit 3; }
done

cp "$BASE" "$TMP/base.yaml"
sed 's/ BEAM3R LINE2 / SOLID LINE2 /' "$BASE" > "$TMP/solid.yaml"
sed 's/ BEAM3R LINE2 / WALL LINE2 /'  "$BASE" > "$TMP/wall.yaml"

probe BEAM3R "$TMP/base.yaml"
probe SOLID  "$TMP/solid.yaml"
probe WALL   "$TMP/wall.yaml"

# The control is a working beam mesh.
grep -m1 -F "processor 0 finished normally" "$TMP/BEAM3R.log"
echo "BASE_BEAM_ELEMENTS=$(grep -c 'BEAM3R LINE2' "$TMP/base.yaml")"

# The element type is recognised; the cell type is not.
grep -m1 -F "Element 'SOLID' does not seem to know cell type 'line2'." "$TMP/SOLID.log"
grep -m1 -F "Element 'WALL' does not seem to know cell type 'line2'." "$TMP/WALL.log"
grep -m1 -F "4C_fem_general_element_definition.cpp" "$TMP/SOLID.log"
grep -m1 -F "4C_fem_general_element_definition.cpp" "$TMP/WALL.log"

# The old attribution is not what happens.
python3 - "$TMP/SOLID.log" "$TMP/WALL.log" <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += (t.count("parobjectfactory")
          + t.count("unknown type 'solid'")
          + t.count("unknown type 'wall'"))
print("CLAIMED_PAROBJECTFACTORY_TEXTS=%d" % n)
PY
exit 0
