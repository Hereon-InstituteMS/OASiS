#!/bin/bash
# Tier-2 for fourc::shell#0 — Kirchhoff-Love shells in 4C, and the three names
# that do NOT work.
#
# The rule (C1 continuity, so NURBS) is right.  The names in the entry were not:
#   * the element is SHELL_KIRCHHOFF_LOVE_NURBS, not "SHELL_KL_NURBS";
#   * it exists for the NURBS9 cell only — pairing it with QUAD4 does not
#     produce a cell-type message, it makes the ELEMENT TYPE itself unknown,
#     which sends you looking in the wrong place;
#   * there is no DKT element in this build at all;
#   * the literal "SHELL KIRCHHOFF QUAD4" spelling is parsed as element type
#     SHELL with cell type KIRCHHOFF and dies on the cell type;
#   * its material is MAT_Kirchhoff_Love_shell (YOUNG_MODULUS / POISSON_RATIO /
#     THICKNESS — thickness lives in the MATERIAL here, not on the element).
#
# Baseline is the upstream deck shell_kl_nurbs, which runs as shipped.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream shell_kl_nurbs.4C.yaml) || exit 3
grep -q "SHELL_KIRCHHOFF_LOVE_NURBS NURBS9" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/good.4C.yaml"
sed 's/SHELL_KIRCHHOFF_LOVE_NURBS NURBS9/SHELL_KL_NURBS NURBS9/'             "$BASE" > "$TMP/shortname.4C.yaml"
sed 's/SHELL_KIRCHHOFF_LOVE_NURBS NURBS9/SHELL_KIRCHHOFF_LOVE_NURBS QUAD4/'  "$BASE" > "$TMP/quad4.4C.yaml"
sed 's/SHELL_KIRCHHOFF_LOVE_NURBS NURBS9/DKT TRI3/'                          "$BASE" > "$TMP/dkt.4C.yaml"
sed 's/SHELL_KIRCHHOFF_LOVE_NURBS NURBS9/SHELL KIRCHHOFF QUAD4/'             "$BASE" > "$TMP/spelled.4C.yaml"

probe GOOD      "$TMP/good.4C.yaml"
probe SHORTNAME "$TMP/shortname.4C.yaml"
probe QUAD4     "$TMP/quad4.4C.yaml"
probe DKT       "$TMP/dkt.4C.yaml"
probe SPELLED   "$TMP/spelled.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
grep -m1 -F "Unknown type 'SHELL_KL_NURBS' of finite element" "$TMP/SHORTNAME.log"
grep -m1 -F "Unknown type 'SHELL_KIRCHHOFF_LOVE_NURBS' of finite element" "$TMP/QUAD4.log"
grep -m1 -F "Unknown type 'DKT' of finite element" "$TMP/DKT.log"
grep -m1 -F "Unknown celltype KIRCHHOFF" "$TMP/SPELLED.log"
grep -m1 -F "4C_comm_parobjectfactory.cpp" "$TMP/DKT.log"
# The QUAD4 arm's real mistake is the CELL type, but the message blames the
# element type and never echoes the cell type it was handed.
echo "QUAD4_DIAGNOSTIC_NAMES_THE_CELL=$(grep -ci 'quad4' "$TMP/QUAD4.log")"
exit 0
