#!/bin/bash
# Tier-2 for fourc::beams#1 — NUMDOF on a beam's DESIGN POINT DIRICH block must
# equal the element's DOF count, and 4C says so in exactly these words:
#
#     3 DOFs given but 6 expected in Point Dirichlet boundary condition
#
# from core/fem/src/discretization/4C_fem_discretization_utils_dbc.cpp.
#
# The entry used to quote 'inconsistent DOF count for beam element'. That string
# is in no 4C source file and never appears in the output; the last assertion
# below keeps that correction pinned.
#
# Upstream deck: beam3r_line2_static_test1 — a 5-element BEAM3R LINE2 cantilever
# whose clamped end carries NUMDOF 6. The bad arm shortens it to 3.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_static_test1.4C.yaml) || exit 3
grep -q "BEAM3R LINE2" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/good.yaml"

python3 - "$BASE" "$TMP/bad.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
good = """  - E: 1
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
"""
bad = """  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
"""
assert good in t, "upstream deck no longer carries the 6-DOF point Dirichlet"
open(sys.argv[2], "w").write(t.replace(good, bad, 1))
PY

probe SIX   "$TMP/good.yaml"
probe THREE "$TMP/bad.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SIX.log"
grep -m1 -F "3 DOFs given but 6 expected in Point Dirichlet boundary condition" "$TMP/THREE.log"
grep -m1 -F "4C_fem_discretization_utils_dbc.cpp" "$TMP/THREE.log"
# The wording the entry used to quote does not exist.
echo "CLAIMED_INCONSISTENT_DOF_TEXT=$(grep -ci 'inconsistent DOF count for beam element' "$TMP/THREE.log")"
# And the failure is at condition read-in, not at element read-in: the mesh was
# already built when it fired.
if grep -q 'fill_complete() on discretization structure' "$TMP/THREE.log"; then
  echo "MESH_WAS_READ_BEFORE_ABORT=yes"
else
  echo "MESH_WAS_READ_BEFORE_ABORT=no"
fi
exit 0
