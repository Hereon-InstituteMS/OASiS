#!/bin/bash

# Tier-2 for fourc::electrochemistry#5, and the execution that CORRECTED it.
#
# The rule holds — S2I coupling needs the dynamics subsection AND the surface
# conditions — but for the opposite reason to the one given, and under
# different section names.
#
#   entry's name  'DESIGN SURF S2I COUPLING CONDITIONS' -> not a section name
#                 on this build at all.  The real ones are
#                 'DESIGN S2I KINETICS <SURF|LINE> CONDITIONS' and
#                 'DESIGN S2I MESHTYING <SURF|LINE> CONDITIONS'.
#   entry's claim omitting the conditions "compiles but never applies the BV
#                 kinetics ... current across the electrode is ~0".  It does
#                 not run at all:
#                 "For each 'S2IKinetics' or 'S2ISCLCoupling' condition a
#                  corresponding 'S2IMeshtying' or 'S2INoEvaluation' condition
#                  has to be defined!"  (src/scatra/4C_scatra_utils.cpp)
#   and dropping SCALAR TRANSPORT DYNAMIC/S2I COUPLING instead gives
#                 "Type of mortar meshtying for scatra-scatra interface
#                  coupling not recognized!"
#
# Neither omission is survivable, so there is no silent ~0 current to look
# for.
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

BASE=$(upstream elch_2D_quad4_s2i_butlervolmer_cycling.4C.yaml) || exit 3
grep -q '^SCALAR TRANSPORT DYNAMIC/S2I COUPLING:$'  "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '^DESIGN S2I KINETICS LINE CONDITIONS:$'    "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" <<'S2IPY'
import sys
from pathlib import Path
t = open(sys.argv[1]).read(); T = Path(sys.argv[2])
i = t.index("DESIGN S2I KINETICS LINE CONDITIONS:")
j = t.index("SCATRA FLUX CALC LINE CONDITIONS:")
(T/"no_kinetics.yaml").write_text(t[:i] + t[j:])
sec = 'SCALAR TRANSPORT DYNAMIC/S2I COUPLING:\n  COUPLINGTYPE: "MatchingNodes"\n'
assert sec in t
(T/"no_s2i_section.yaml").write_text(t.replace(sec, ""))
(T/"claimed_name.yaml").write_text(
    t.replace("DESIGN S2I KINETICS LINE CONDITIONS:",
              "DESIGN SURF S2I COUPLING CONDITIONS:"))
S2IPY

probe BASELINE       "$BASE"
probe NO_KINETICS    "$TMP/no_kinetics.yaml"
probe NO_S2I_SECTION "$TMP/no_s2i_section.yaml"
probe CLAIMED_NAME   "$TMP/claimed_name.yaml"

echo "BASELINE_PASSED=$(grep -c 'is CORRECT' "$TMP/BASELINE.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASELINE.log"

# Dropping the condition block is NOT a quiet loss of the kinetics.
grep -m1 -F "For each 'S2IKinetics' or 'S2ISCLCoupling' condition a corresponding 'S2IMeshtying' or 'S2INoEvaluation' condition has to be defined!" "$TMP/NO_KINETICS.log"
grep -m1 -F "4C_scatra_utils.cpp" "$TMP/NO_KINETICS.log"
# Dropping the dynamics subsection is not quiet either.
grep -m1 -F "Type of mortar meshtying for scatra-scatra interface coupling not recognized!" "$TMP/NO_S2I_SECTION.log"
grep -m1 -F "4C_scatra_timint_meshtying_strategy_s2i.cpp" "$TMP/NO_S2I_SECTION.log"
# The section name the entry gives does not exist on this build.
grep -m1 -F "Section 'DESIGN SURF S2I COUPLING CONDITIONS' is not a valid section name." "$TMP/CLAIMED_NAME.log"

echo "NO_KINETICS_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NO_KINETICS.log")"
echo "NO_S2I_SECTION_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NO_S2I_SECTION.log")"
echo "REAL_CONDITION_SECTIONS=$(grep -cE '^DESIGN S2I (KINETICS|MESHTYING) LINE CONDITIONS:$' "$BASE")"
exit 0
