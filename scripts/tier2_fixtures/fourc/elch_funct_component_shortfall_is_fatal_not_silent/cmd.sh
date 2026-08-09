#!/bin/bash

# Tier-2 for fourc::electrochemistry#3, and the execution that FALSIFIED it.
#
# The entry said a multi-component FUNCT missing one COMPONENT "silently sets
# that scalar to 0 — visible as discontinuous initial concentration plot for
# the missing species".  It is not silent and there is no plot to inspect:
# delete the last COMPONENT of the initial-field FUNCT and 4C aborts on the
# first evaluation with
#
#   "There are 4 expressions but tried to access component 4"
#   from core/utils/src/functions/4C_utils_function.cpp
#
# The run never reaches its result tests.
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

BASE=$(upstream elch_1D_10ele_3ions_stab_fdcheck.4C.yaml) || exit 3

python3 - "$BASE" "$TMP/short_funct.yaml" <<'FPY'
import sys
t = open(sys.argv[1]).read()
tail = """  - COMPONENT: 4
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
RESULT DESCRIPTION:"""
assert tail in t, "upstream deck no longer carries FUNCT2 COMPONENT 4"
open(sys.argv[2], "w").write(t.replace(tail, "RESULT DESCRIPTION:"))
FPY
echo "COMPONENTS_IN_BASELINE=$(grep -c 'COMPONENT:' "$BASE")"
echo "COMPONENTS_IN_SHORT=$(grep -c 'COMPONENT:' "$TMP/short_funct.yaml")"

probe BASELINE    "$BASE"
probe SHORT_FUNCT "$TMP/short_funct.yaml"

echo "BASELINE_PASSED=$(grep -c 'is CORRECT' "$TMP/BASELINE.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASELINE.log"

# The missing component is NOT quietly defaulted to zero.
grep -m1 -F "There are 4 expressions but tried to access component 4" "$TMP/SHORT_FUNCT.log"
grep -m1 -F "4C_utils_function.cpp" "$TMP/SHORT_FUNCT.log"
echo "SHORT_FUNCT_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SHORT_FUNCT.log")"
if grep -q "There are 4 expressions but tried to access component 4" "$TMP/SHORT_FUNCT.log"; then
  echo "MISSING_COMPONENT_IS_SILENTLY_ZERO=no"
else
  echo "MISSING_COMPONENT_IS_SILENTLY_ZERO=yes"
fi
exit 0
