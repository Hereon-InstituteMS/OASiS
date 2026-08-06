#!/bin/bash

# Tier-2 for fourc::electrochemistry#4 — CALCFLUX_DOMAIN: 'total' is what puts
# species fluxes into the VTK output, and its absence is an ABSENCE: the
# arrays are simply not in the dataset.
#
#   without the key -> .vtu carries phi_1 ... phi_4 and nothing else
#   with 'total'    -> .vtu additionally carries flux_domain_phi_1 ... _phi_4
#
# The solution itself is untouched: both arms pass all ten result tests.  The
# entry's Signal said the flux fields "show 'not computed'"; that string
# appears neither in the log nor in the .vtu, because there is no field there
# to say anything.
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

BASE=$(upstream elch_2D_tertiary_twoEqu_ENC_varParams_ndb_2iter.4C.yaml) || exit 3
grep -q '^  SOLVERTYPE: "nonlinear"$' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
echo "BASE_DECK_HAS_CALCFLUX=$(grep -c 'CALCFLUX_DOMAIN' "$BASE")"

cp "$BASE" "$TMP/noflux.yaml"
sed 's/^  SOLVERTYPE: "nonlinear"$/  SOLVERTYPE: "nonlinear"\n  CALCFLUX_DOMAIN: "total"/' \
    "$BASE" > "$TMP/flux.yaml"

probe NOFLUX "$TMP/noflux.yaml"
probe FLUX   "$TMP/flux.yaml"

echo "NOFLUX_PASSED=$(grep -c 'is CORRECT' "$TMP/NOFLUX.log")"
echo "FLUX_PASSED=$(grep -c 'is CORRECT' "$TMP/FLUX.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/FLUX.log"

arrays() { grep -oh 'Name="[^"]*"' "$TMP/o_$1-vtk-files"/*.vtu 2>/dev/null | sort -u | tr '\n' ' '; }
echo "NOFLUX_ARRAYS= $(arrays NOFLUX)"
echo "FLUX_ARRAYS= $(arrays FLUX)"
echo "WITHOUT_KEY_FLUX_ARRAYS=$(arrays NOFLUX | grep -o 'flux_domain_phi_[0-9]*' | wc -l)"
echo "WITH_TOTAL_FLUX_ARRAYS=$(arrays FLUX   | grep -o 'flux_domain_phi_[0-9]*' | wc -l)"

# The entry claimed the flux fields "show 'not computed'".  They do not show
# anything: the arrays are simply absent from the dataset, and that string is
# in neither log nor either .vtu.
echo "CLAIMED_NOT_COMPUTED_TEXT=$(cat "$TMP"/NOFLUX.log "$TMP"/o_NOFLUX-vtk-files/*.vtu 2>/dev/null | grep -ci 'not computed')"
exit 0
