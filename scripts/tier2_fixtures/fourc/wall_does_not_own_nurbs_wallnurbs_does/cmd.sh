#!/bin/bash
# Tier-2 for fourc::structural_mechanics#2 — the isogeometric 2D cell is
# WALLNURBS, not WALL, and the message you get for the wrong spelling is
# misleading.
#
# The upstream deck w1_w1nurbs.4C.yaml is the ideal witness because it carries
# BOTH families in one mesh: four "WALLNURBS NURBS9" cells on CP control points
# and four "WALL QUAD4" cells on plain NODE lines.  It runs to completion, so
# WALL is demonstrably a registered element type in the very same run in which
#
#     "WALL NURBS9 ..."  ->  Unknown type 'WALL' of finite element
#
# is thrown.  What is unregistered is the WALL-plus-NURBS pairing, not WALL.
#
# The third arm is the dangerous one: NURBS cells need their geometry written as
# 'CP <id> COORD x y z <weight>'.  Rewriting exactly those lines as ordinary
# 'NODE <id> COORD x y z' — the whole deck otherwise untouched, SHAPEFCT and
# KNOTVECTORS still in place — does not produce a diagnostic.  It SEGFAULTS
# inside Wall1::w1_nlnstiffmass with shell status 139 and no 4C error block.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream w1_w1nurbs.4C.yaml) || exit 3

# If upstream ever renames these, the fixture must break loudly rather than
# quietly test nothing.
for needle in 'WALLNURBS NURBS9' 'SHAPEFCT: "Nurbs"' 'STRUCTURE KNOTVECTORS:' '"CP 1 COORD'; do
  grep -qF "$needle" "$BASE" || {
    echo "FIXTURE_ABORT=upstream_deck_changed (missing: $needle)"; exit 3; }
done

cp "$BASE" "$TMP/base.yaml"
sed 's/WALLNURBS NURBS9/WALL NURBS9/' "$BASE" > "$TMP/wall_nurbs.yaml"
python3 - "$BASE" "$TMP/node_lines.yaml" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
out = re.sub(r'"CP (\d+) COORD (\S+) (\S+) (\S+) (\S+)"',
             r'"NODE \1 COORD \2 \3 \4"', src)
assert '"CP ' not in out
open(sys.argv[2], "w").write(out)
PY

probe BASE      "$TMP/base.yaml"
probe WALLNURBS "$TMP/wall_nurbs.yaml"
probe NODELINES "$TMP/node_lines.yaml"

# The control runs, and it contains WALL elements — so WALL is a known type.
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_HAS_PLAIN_WALL_ELEMENTS=$(grep -c 'WALL QUAD4' "$TMP/base.yaml")"

# The wrong spelling blames the element type that in fact exists.
grep -m1 -F "Unknown type 'WALL' of finite element" "$TMP/WALLNURBS.log"
grep -m1 -F "4C_comm_parobjectfactory.cpp" "$TMP/WALLNURBS.log"

# Plain NODE lines under a NURBS cell: no diagnostic at all, just a signal.
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/NODELINES.log"
echo "NODELINES_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/NODELINES.log")"

# The three extra requirements the entry lists are all in the working deck.
python3 - "$TMP/base.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
print("NURBS_DECK_HAS_SHAPEFCT=%s" % ("yes" if 'SHAPEFCT: "Nurbs"' in t else "no"))
print("NURBS_DECK_HAS_KNOTVECTORS=%s"
      % ("yes" if "STRUCTURE KNOTVECTORS:" in t else "no"))
print("NURBS_DECK_CP_LINES=%d" % t.count('"CP '))
PY
exit 0
