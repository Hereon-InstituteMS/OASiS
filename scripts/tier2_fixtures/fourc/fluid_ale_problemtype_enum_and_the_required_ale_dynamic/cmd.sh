#!/bin/bash

# Tier-2 for fourc::fluid#2 — 'Fluid_Ale' is the enum spelling, ALE DYNAMIC
# really is required, and picking the wrong problem type for an ALE deck is
# the worst of the three failures.
#
# Baseline is the upstream deck f2_loc_sys_bc_ale.4C.yaml, mutated in one
# place per arm:
#
#   BASE   untouched                    -> exit 0, four result tests CORRECT
#   NOALE  the ALE DYNAMIC block deleted -> exit 1, and the abort asks for
#                                          LINEAR_SOLVER in ALE DYNAMIC by name
#   CASE   PROBLEMTYPE "Fluid_ALE"       -> exit 1, and 4C prints the whole
#                                          legal enum, which contains Fluid_Ale
#   PLAIN  PROBLEMTYPE "Fluid"           -> SIGSEGV, exit 139, no diagnostic of
#                                          any kind and no result test reached
# --- self-contained preamble (deliberately NOT sourced from ../_lib) --------
# scripts/mutate_tier2_fixtures.py stages this directory into a scratch tree.
# Everything the fixture needs is inline, so the mutation proof cannot be
# confounded by a missing sibling.  Same honesty rule as the shared preamble:
# when 4C is missing this prints FIXTURE_ABORT=no_binary and exits non-zero,
# and fixture.json forbids both strings, so an absent solver makes the fixture
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
# ---------------------------------------------------------------------------

# Locate 4C's own regression decks.  Missing decks abort loudly, same rule as
# a missing binary, and fixture.json forbids the marker.
DECKS=""
for _d in "${FOURC_INPUT_FILES:-}" "$HOME/4C/tests/input_files" \
          "$HOME/Schreibtisch/4C-src/4C/tests/input_files"; do
  [ -n "${_d:-}" ] && [ -d "$_d" ] && DECKS="$_d" && break
done
upstream() {
  if [ -z "$DECKS" ] || [ ! -f "$DECKS/$1" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (missing $1)"
    exit 3
  fi
  printf '%s\n' "$DECKS/$1"
}

BASE=$(upstream f2_loc_sys_bc_ale.4C.yaml) || exit 3
grep -q '^  PROBLEMTYPE: "Fluid_Ale"$' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '^ALE DYNAMIC:$'                 "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '^FLUID DYNAMIC:$'               "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/base.yaml"
python3 - "$BASE" "$TMP/noale.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
i = t.index("ALE DYNAMIC:\n")
j = t.index("FLUID DYNAMIC:\n")
open(sys.argv[2], "w").write(t[:i] + t[j:])
PY
sed 's|^  PROBLEMTYPE: "Fluid_Ale"|  PROBLEMTYPE: "Fluid_ALE"|' "$BASE" > "$TMP/case.yaml"
sed 's|^  PROBLEMTYPE: "Fluid_Ale"|  PROBLEMTYPE: "Fluid"|'     "$BASE" > "$TMP/plain.yaml"

# The deletion has to be real.
echo "NOALE_DECK_HAS_ALE_DYNAMIC=$(grep -c '^ALE DYNAMIC:$' "$TMP/noale.yaml")"

probe BASE  "$TMP/base.yaml"
probe NOALE "$TMP/noale.yaml"
probe CASE  "$TMP/case.yaml"
probe PLAIN "$TMP/plain.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BASE.log")"
echo "BASE_RESULT_PASSES=$(grep -c 'is CORRECT' "$TMP/BASE.log")"

# ALE DYNAMIC is required, and the message names the key inside it.
grep -m1 -F "No linear solver defined for ALE problems. Please set LINEAR_SOLVER in ALE DYNAMIC to a valid number!" "$TMP/NOALE.log"
grep -m1 -F "4C_adapter_ale.cpp" "$TMP/NOALE.log"

# The enum spelling is exact, and 4C prints the alternatives when it is not.
grep -m1 -F "Candidate deprecated_selection 'PROBLEMTYPE' has wrong value" "$TMP/CASE.log"
echo "ENUM_LIST_CONTAINS_FLUID_ALE=$(grep -c '|Fluid_Ale|' "$TMP/CASE.log")"

# Running an ALE deck as a plain fluid problem is not diagnosed at all.
echo "PLAIN_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/PLAIN.log")"
echo "PLAIN_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/PLAIN.log")"
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/PLAIN.log"
exit 0
