#!/bin/bash
# Tier-2 for fourc::fbi#1 — the penalty parameter really does have to be tuned,
# but the entry names the WRONG SECTION for it and quotes two diagnostics 4C
# never prints.
#
# Section path.  The entry says "BEAM INTERACTION/BEAM TO FLUID MESHTYING
# PENALTY_PARAMETER".  That section does not exist; 4C rejects it outright:
#   "Section 'BEAM INTERACTION/BEAM TO FLUID MESHTYING' is not a valid section
#    name." from core/io/src/4C_io_input_file.cpp line 546.
# The real path is FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING.
#
# Both bounds, on upstream fbi_mortar_solidcoupling.4C.yaml (PENALTY_PARAMETER
# 0.1, six result tests):
#   1e+14 -> "The nonlinear solver did not converge!" from
#            solver_nonlin_nox/4C_solver_nonlin_nox_problem.cpp line 165.
#            Not a linear-solver conditioning message: nothing mentions cond(K).
#   1e-10 -> runs clean, exits only because 2 of the 6 result tests moved; the
#            beam is under-constrained and simply does not pick up the load.
#            Nothing in the log says 'slip through' or anything like it.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fbi_mortar_solidcoupling.4C.yaml) || exit 3
grep -q '  PENALTY_PARAMETER: 0.1' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_penalty_changed"; exit 3; }
grep -q '^FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_section_path_changed"; exit 3; }
cp "$(dirname "$BASE")/beam_flow_solver.xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_nox_xml"; exit 3; }
cd "$TMP" || exit 3

# The three pathologies.
HUGE_PENALTY=1e+14
TINY_PENALTY=1e-10
CLAIMED_SECTION='BEAM INTERACTION/BEAM TO FLUID MESHTYING:'

cp "$BASE" "$TMP/tuned.yaml"
sed "s/  PENALTY_PARAMETER: 0.1/  PENALTY_PARAMETER: $HUGE_PENALTY/" "$BASE" > "$TMP/huge.yaml"
sed "s/  PENALTY_PARAMETER: 0.1/  PENALTY_PARAMETER: $TINY_PENALTY/" "$BASE" > "$TMP/tiny.yaml"
sed "s|^FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING:|$CLAIMED_SECTION|" \
    "$BASE" > "$TMP/claimedsec.yaml"

probe TUNED      "$TMP/tuned.yaml"
probe HUGE       "$TMP/huge.yaml"
probe TINY       "$TMP/tiny.yaml"
probe CLAIMEDSEC "$TMP/claimedsec.yaml"

grep -m1 -F "OK (6)" "$TMP/TUNED.log"
grep -m1 -F "processor 0 finished normally" "$TMP/TUNED.log"
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/HUGE.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/HUGE.log"
grep -m1 -F "Result check failed with 2 errors out of 6 tests" "$TMP/TINY.log"
grep -m1 -F "Section 'BEAM INTERACTION/BEAM TO FLUID MESHTYING' is not a valid section name." "$TMP/CLAIMEDSEC.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/CLAIMEDSEC.log"

# The two quoted diagnostics are absent.
echo "CLAIMED_CONDITION_NUMBER_TEXT=$(grep -ciE 'cond\(K\)|condition number' "$TMP/HUGE.log")"
echo "CLAIMED_SLIP_THROUGH_TEXT=$(grep -ci 'slip through' "$TMP/TINY.log")"
# The small-penalty run is the silent one: it converges and completes.
echo "TINY_PENALTY_NONCONVERGENCE=$(grep -c 'did not converge' "$TMP/TINY.log")"
echo "TINY_PENALTY_WARNINGS=$(grep -ciE 'penalty.*(small|large|tune|range)' "$TMP/TINY.log")"
echo "TINY_PENALTY_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/TINY.log")"
echo "TUNED_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/TUNED.log")"
exit 0
