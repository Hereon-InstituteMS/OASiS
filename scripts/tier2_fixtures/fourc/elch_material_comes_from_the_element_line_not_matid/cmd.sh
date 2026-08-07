#!/bin/bash

# Tier-2 for fourc::electrochemistry#1, and the execution that FALSIFIED it.
#
# The entry said "MATID in SCALAR TRANSPORT DYNAMIC must reference the
# MAT_matlist material" and that pointing it at a MAT_ion aborts with
# "expected matlist, got ion" from 4C_scatra_factory.cpp.  Measured:
#
#   MATID: 3 (a MAT_ion)  -> runs, all ten result tests pass
#   MATID deleted         -> runs, all ten result tests pass
#
# MATID in that section is inert here; 4C documents it as the material for
# automatic mesh generation.  What the rule is really about is the ELEMENT
# LINE's "MAT <id>", and repointing THAT at a MAT_ion does abort — with a
# different message, from a different file:
#
#   "Invalid material type!"  from
#   src/scatra_ele/4C_scatra_ele_calc_service_elch_NP.cpp, in
#   ScaTraEleCalcElchNP::check_elch_element_parameter
#
# Neither the quoted text nor the file 4C_scatra_factory.cpp exists.
# --- self-contained preamble (deliberately NOT sourced from ../_lib) --------
# scripts/mutate_tier2_fixtures.py copies ONLY this directory into a scratch
# tree.  A fixture that sources ../_lib/preamble.sh therefore cannot even
# start there, its mutant dies for the wrong reason, and the KILLED verdict
# certifies nothing.  Everything this fixture needs is inline, so the
# mutation proof is real.  Same honesty rule as the shared preamble: when 4C
# is missing this prints FIXTURE_ABORT=no_binary and exits non-zero, and
# fixture.json forbids both strings, so an absent solver makes the fixture
# RED rather than green.
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" \
          "$HOME/Schreibtisch/4C-src/4C/build/4C" "/usr/local/bin/4C"; do
  [ -n "${_c:-}" ] && [ -x "$_c" ] && BIN="$_c" && break
done
if [ -z "${BIN:-}" ]; then
  echo "FIXTURE_ABORT=no_binary (set FOURC_BINARY to a 4C executable)"
  exit 3
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# stdbuf is not decoration: 4C writes result-test verdicts to raw std::cout
# and MPI_Abort discards a block-buffered stdout (pitfall input_format#18).
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }
probe() { run4c "$2" "$TMP/o_$1" > "$TMP/$1.log" 2>&1; echo "EXIT_$1=$?"; }
DECKS=""
for _d in "${FOURC_INPUT_FILES:-}" "$HOME/4C/tests/input_files" \
          "$HOME/Schreibtisch/4C-src/4C/tests/input_files"; do
  [ -n "${_d:-}" ] && [ -d "$_d" ] && DECKS="$_d" && break
done
upstream() {  # <name.4C.yaml> -> its path, or abort loudly
  if [ -z "$DECKS" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (set FOURC_INPUT_FILES)"; exit 3; fi
  if [ ! -f "$DECKS/$1" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (missing $1)"; exit 3; fi
  printf '%s\n' "$DECKS/$1"
}
# ---------------------------------------------------------------------------

TERT=$(upstream elch_2D_tertiary_twoEqu_ENC_varParams_ndb_2iter.4C.yaml) || exit 3
ION=$(upstream elch_1D_10ele_3ions_stab_fdcheck.4C.yaml) || exit 3
grep -q '^  MATID: 1$'      "$TERT" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'MAT 4 TYPE ElchNP' "$ION"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# MATID in SCALAR TRANSPORT DYNAMIC: point it at a MAT_ion, then delete it.
sed 's/^  MATID: 1$/  MATID: 3/' "$TERT" > "$TMP/matid_ion.yaml"
grep -v '^  MATID: 1$'           "$TERT" > "$TMP/matid_absent.yaml"
# The element line's MAT: point it at a MAT_ion instead of the MAT_matlist.
sed 's/MAT 4 TYPE ElchNP/MAT 1 TYPE ElchNP/' "$ION" > "$TMP/element_mat_ion.yaml"
echo "ELEMENT_LINES_REPOINTED=$(grep -c 'MAT 1 TYPE ElchNP' "$TMP/element_mat_ion.yaml")"

probe TERT_BASELINE   "$TERT"
probe MATID_ION       "$TMP/matid_ion.yaml"
probe MATID_ABSENT    "$TMP/matid_absent.yaml"
probe ION_BASELINE    "$ION"
probe ELEMENT_MAT_ION "$TMP/element_mat_ion.yaml"

echo "TERT_BASELINE_PASSED=$(grep -c 'is CORRECT' "$TMP/TERT_BASELINE.log")"
echo "MATID_ION_PASSED=$(grep -c 'is CORRECT' "$TMP/MATID_ION.log")"
echo "MATID_ABSENT_PASSED=$(grep -c 'is CORRECT' "$TMP/MATID_ABSENT.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/MATID_ION.log"

# The element line, by contrast, is load-bearing.
grep -m1 -F "Invalid material type!" "$TMP/ELEMENT_MAT_ION.log"
grep -m1 -F "4C_scatra_ele_calc_service_elch_NP.cpp" "$TMP/ELEMENT_MAT_ION.log"
grep -m1 -o "check_elch_element_parameter" "$TMP/ELEMENT_MAT_ION.log"

# Neither the quoted message nor the file it was attributed to exists.
echo "CLAIMED_EXPECTED_MATLIST_TEXT=$(grep -ci 'expected matlist' "$TMP/ELEMENT_MAT_ION.log")"
echo "CLAIMED_SCATRA_FACTORY_FILE=$(grep -c '4C_scatra_factory.cpp' "$TMP/ELEMENT_MAT_ION.log")"
exit 0
