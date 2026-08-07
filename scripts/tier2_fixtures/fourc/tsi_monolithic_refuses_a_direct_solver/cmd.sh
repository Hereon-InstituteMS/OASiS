#!/bin/bash
# Tier-2 for fourc::tsi#8 — monolithic TSI refuses a direct solver.  Upstream
# tsi_lincompression_monolithic run twice; only SOLVER 2 (the one TSI
# DYNAMIC/MONOLITHIC points LINEAR_SOLVER at) differs.
#
#   Belos + Teko block preconditioner -> runs, all result tests CORRECT, exit 0
#   UMFPACK                           -> aborts at TSI::Monolithic::create_linear_solver
#
# The rule holds.  The Signal does not: the claim quoted 'monolithic TSI
# requires Belos' from 4C_tsi_monolithic.cpp.  The file is right, the words are
# not — what 4C prints is the two-word
#
#     Iterative solver expected
#
# which names neither Belos nor TSI nor the solver you actually wrote, so
# grepping the error for "Belos" or for "UMFPACK" finds nothing.  Both the real
# text and the absence of the claimed text are asserted.
#
# The upstream deck's SOLVER_XML_FILE / TEKO_XML_FILE paths are relative to the
# working directory, so both arms run with cwd set to a scratch dir carrying a
# symlink to the upstream xml/ tree.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_lincompression_monolithic.4C.yaml) || exit 3
XMLDIR="$(dirname "$BASE")/xml"
[ -d "$XMLDIR" ] || { echo "FIXTURE_ABORT=no_upstream_decks (missing xml/)"; exit 3; }
ln -s "$XMLDIR" "$TMP/xml"

BELOS_BLOCK='SOLVER 2:
  SOLVER: "Belos"
  AZPREC: "Teko"
  AZREUSE: 10
  SOLVER_XML_FILE: "xml/linear_solver/iterative_gmres_template.xml"
  TEKO_XML_FILE: "xml/block_preconditioner/thermo_solid.xml"
  NAME: "Thermo_Structure_Interaction_Solver"
'
python3 - "$BASE" "$TMP/umfpack.yaml" "$BELOS_BLOCK" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = sys.argv[3]
if old not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
new = 'SOLVER 2:\n  SOLVER: "UMFPACK"\n  NAME: "Thermo_Structure_Interaction_Solver"\n'
open(sys.argv[2], "w").write(t.replace(old, new))
PY
[ -s "$TMP/umfpack.yaml" ] || exit 3
cp "$BASE" "$TMP/belos.yaml"

mono() {  # $1 = label, $2 = deck
  ( cd "$TMP" && stdbuf -oL -eL "$BIN" "$2" "$TMP/o_$1" > "$TMP/$1.log" 2>&1 )
  echo "EXIT_$1=$?"
}

mono BELOS   "$TMP/belos.yaml"
mono UMFPACK "$TMP/umfpack.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BELOS.log"
echo "BELOS_RESULT_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BELOS.log")"
grep -m1 -F "Iterative solver expected" "$TMP/UMFPACK.log"
grep -m1 -oF "4C_tsi_monolithic.cpp" "$TMP/UMFPACK.log"
grep -m1 -oF "TSI::Monolithic::create_linear_solver" "$TMP/UMFPACK.log"
# The catalogued wording does not exist, and the message names nothing useful.
echo "CLAIMED_REQUIRES_BELOS_TEXT=$(grep -ci 'monolithic TSI requires Belos' "$TMP/UMFPACK.log")"
echo "DIAGNOSTIC_NAMES_BELOS=$(grep -c 'Iterative solver expected.*Belos' "$TMP/UMFPACK.log")"
echo "DIAGNOSTIC_NAMES_UMFPACK=$(grep -c 'Iterative solver expected.*UMFPACK' "$TMP/UMFPACK.log")"
# It never reaches a time step.
echo "UMFPACK_REACHED_FIRST_TIME_STEP=$(grep -c '^TIME: ' "$TMP/UMFPACK.log")"
exit 0
