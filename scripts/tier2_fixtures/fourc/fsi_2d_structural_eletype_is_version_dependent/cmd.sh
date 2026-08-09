#!/bin/bash
# Tier-2 for fourc::fsi#13 — WHICH eletype owns 2D structural cells inside an FSI
# deck is VERSION-DEPENDENT, and the entry states one era as if it were the only
# one.  The two spellings share no keywords:
#
#   4C <= 2026.2:  WALL  QUAD4 <n..> MAT m KINEM k EAS none THICK t
#                        STRESS_STRAIN s GP 2 2
#   4C >= 2026.3:  SOLID QUAD4 <n..> MAT m KINEM k THICKNESS t
#                        PLANE_ASSUMPTION p
#
# The entry asserts the SOLID era ("4C 2026.3 2D structural element name is
# 'SOLID QUAD4' (NOT 'WALL QUAD4')" ... "Real syntax in tests/input_files/
# contact2D_*.4C.yaml") and reports the loser's message as
# "Unknown type WALL of finite element" from 4C_comm_parobjectfactory.cpp:153.
#
# On the deployed binary the era is the OTHER one, and so is the message.  Rather
# than hard-code either, this fixture takes the upstream 2D FSI deck
# volmortar2D_fsi.4C.yaml, rewrites its STRUCTURE ELEMENTS into BOTH spellings,
# runs both, and asserts the invariant that holds in either era: exactly one of
# the two factories accepts quad4, the winner reaches fill_complete and matches
# the deck's pinned results, and the loser is rejected by an element-vocabulary
# diagnostic.  It then prints a machine-readable verdict naming the winner.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream volmortar2D_fsi.4C.yaml) || exit 3
grep -q '^  DIM: 2' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_is_not_2d"; exit 3; }
grep -q 'PROBLEMTYPE: "Fluid_Structure_Interaction"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_is_not_fsi"; exit 3; }

# The pathology: write the 2D structural elements with the eletype the deployed
# binary does not own.  Which one that is, is what the fixture measures.
PROBE_BOTH_SPELLINGS=yes

python3 - "$BASE" "$TMP" "$PROBE_BOTH_SPELLINGS" <<'PY'
import re, sys
src, tmp, both = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(src).read()
wall = re.compile(r'"(\d+) WALL QUAD4 ((?:\d+ )+)MAT (\d+) KINEM (\w+) EAS none '
                  r'THICK ([\d.]+) STRESS_STRAIN (\w+) GP 2 2"')
solid = re.compile(r'"(\d+) SOLID QUAD4 ((?:\d+ )+)MAT (\d+) KINEM (\w+) '
                   r'THICKNESS ([\d.]+) PLANE_ASSUMPTION (\w+)"')
if wall.search(t):
    as_wall = t
    as_solid = wall.sub(r'"\1 SOLID QUAD4 \2MAT \3 KINEM \4 THICKNESS \5 '
                        r'PLANE_ASSUMPTION \6"', t)
elif solid.search(t):
    as_solid = t
    as_wall = solid.sub(r'"\1 WALL QUAD4 \2MAT \3 KINEM \4 EAS none THICK \5 '
                        r'STRESS_STRAIN \6 GP 2 2"', t)
else:
    print("FIXTURE_ABORT=upstream_2d_structure_elements_in_neither_spelling")
    sys.exit(3)
if both != "yes":
    as_solid = as_wall
open(tmp + "/wall.yaml", "w").write(as_wall)
open(tmp + "/solid.yaml", "w").write(as_solid)
print("WALL_ARM_ELEMENT_LINES=%d" % as_wall.count("WALL QUAD4"))
print("SOLID_ARM_ELEMENT_LINES=%d" % as_solid.count("SOLID QUAD4"))
PY
[ -f "$TMP/wall.yaml" ] || exit 3

probe WALL  "$TMP/wall.yaml"
probe SOLID "$TMP/solid.yaml"

acc() {  # a spelling is "accepted" if its structure discretization was built
  grep -c 'fill_complete() on discretization structure' "$TMP/$1.log"
}
W=$(acc WALL); S=$(acc SOLID)
echo "WALL_REACHED_FILL_COMPLETE=$([ "$W" -gt 0 ] && echo 1 || echo 0)"
echo "SOLID_REACHED_FILL_COMPLETE=$([ "$S" -gt 0 ] && echo 1 || echo 0)"

if { [ "$W" -gt 0 ] && [ "$S" -eq 0 ]; } || { [ "$S" -gt 0 ] && [ "$W" -eq 0 ]; }; then
  echo "EXACTLY_ONE_2D_FSI_FACTORY=yes"
else
  echo "EXACTLY_ONE_2D_FSI_FACTORY=no"
fi
if [ "$W" -gt 0 ]; then WINNER=WALL; LOSER=SOLID; else WINNER=SOLID; LOSER=WALL; fi
echo "VERDICT: FSI_2D_STRUCT_ELETYPE=$WINNER"

# The winner is a working FSI deck, not just a parse.
grep -m1 -F "fill_complete() on discretization structure" "$TMP/$WINNER.log"
grep -m1 -F "OK (6)" "$TMP/$WINNER.log"
grep -m1 -F "processor 0 finished normally" "$TMP/$WINNER.log"

# The loser is rejected by an element-vocabulary diagnostic, whichever era we
# are in; print whichever sentence this binary really used.
grep -m1 -iE "does not seem to know cell type|Unknown type .* of finite element" "$TMP/$LOSER.log"
if grep -qiE "does not seem to know cell type|Unknown type .* of finite element" "$TMP/$LOSER.log"; then
  echo "LOSER_DIAGNOSTIC_IS_A_CELL_TYPE_REJECTION=yes"
else
  echo "LOSER_DIAGNOSTIC_IS_A_CELL_TYPE_REJECTION=no"
fi
echo "LOSER_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/$LOSER.log")"
exit 0
