#!/bin/bash

# Tier-2 for fourc::electrochemistry#2 — the number of transported scalars is
# the matlist's NUMMAT (one per ionic species) PLUS one for the electric
# potential, and 4C states the expected number itself when a block gets it
# wrong.
#
# The upstream deck carries MAT_matlist NUMMAT 4 and every boundary condition
# declares NUMDOF 5.  Cut one DOF off the point Dirichlet block and:
#
#   "4 DOFs given but 5 expected in Point Dirichlet boundary condition"
#   from core/fem/src/discretization/4C_fem_discretization_utils_dbc.cpp
#
# That also FALSIFIES the entry's Signal on both counts: the failure is not
# silent (the run never reaches its result tests) and the phrasing it quoted,
# 'INITIALFIELD component count mismatch', does not exist.
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
grep -q 'NUMMAT: 4' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# The deck's matlist holds 4 ionic species, and every boundary condition
# declares 5 DOFs: one per species plus one for the electric potential.
echo "MATLIST_NUMMAT=$(grep -oP 'NUMMAT: \K[0-9]+' "$BASE" | head -1)"
echo "DIRICH_NUMDOF=$(grep -oP 'NUMDOF: \K[0-9]+' "$BASE" | head -1)"

python3 - "$BASE" "$TMP/short.yaml" <<'SHORTPY'
import sys
t = open(sys.argv[1]).read()
block = """  - E: 1
    NUMDOF: 5
    ONOFF: [1, 1, 1, 1, 1]
    VAL: [1, 2, 2, 1, 0]
    FUNCT: [0, 0, 0, 0, 0]"""
assert block in t, "upstream deck no longer carries the 5-DOF point Dirichlet block"
t = t.replace(block, """  - E: 1
    NUMDOF: 4
    ONOFF: [1, 1, 1, 1]
    VAL: [1, 2, 2, 1]
    FUNCT: [0, 0, 0, 0]""")
open(sys.argv[2], "w").write(t)
SHORTPY

probe BASELINE "$BASE"
probe SHORT    "$TMP/short.yaml"

echo "BASELINE_PASSED=$(grep -c 'is CORRECT' "$TMP/BASELINE.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASELINE.log"

# 4C states the expected count itself, which is the whole arithmetic.
grep -m1 -F "4 DOFs given but 5 expected in Point Dirichlet boundary condition" "$TMP/SHORT.log"
grep -m1 -F "4C_fem_discretization_utils_dbc.cpp" "$TMP/SHORT.log"

# It is not silent, and the quoted phrasing does not exist.
echo "SHORT_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SHORT.log")"
echo "CLAIMED_INITIALFIELD_COUNT_MISMATCH=$(grep -ci 'INITIALFIELD component count mismatch' "$TMP/SHORT.log")"
exit 0
