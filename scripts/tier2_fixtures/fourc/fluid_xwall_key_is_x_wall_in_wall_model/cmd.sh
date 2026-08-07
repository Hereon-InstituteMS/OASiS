#!/bin/bash

# Tier-2 for fourc::fluid#3, and a FALSIFICATION of the key names and of the
# Signal.
#
# The entry says x-wall is "activated via FLUID DYNAMIC/WALL_NORMAL_NODE_
# DISTANCE and related XWALL_* keys".  Neither exists: grep the whole 4C tree
# and WALL_NORMAL_NODE_DISTANCE is nowhere, while the real switch is the bool
# X_WALL inside the section 'FLUID DYNAMIC/WALL MODEL'.
#
# Baseline is the upstream turbulent-channel deck f3_cha_xwall_6x8x6.4C.yaml,
# which uses the real key:
#
#   BASE      untouched                     -> exit 0, sixteen result tests
#                                              CORRECT
#   OFF       X_WALL: true -> false          -> SIGABRT, exit 134.  X_WALL is
#                                              read at mesh-reading time to
#                                              decide the ELEMENT TYPE, so
#                                              turning it off leaves elements
#                                              asking for a parameter that is
#                                              no longer there:
#                                              Teuchos::Exceptions::Invalid-
#                                              ParameterName, "xwalltoggle"
#   CLAIMKEY  WALL_NORMAL_NODE_DISTANCE      -> exit 1, unmatched key
#   XWALLKEY  XWALL_TAUW_TYPE in WALL MODEL  -> exit 1, unmatched key
#
# The claimed Signal — a log-law slope agreeing to ~5 percent with the model on
# and diverging with it off — is not observable: with the flag off there is no
# velocity profile to compare, the process dies during setup.
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

# 4C resolves IFPACK_XML_FILE relative to the INPUT FILE's directory, so a
# copied deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream f3_cha_xwall_6x8x6.4C.yaml) || exit 3
grep -q '^  X_WALL: true$'                  "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '^FLUID DYNAMIC/WALL MODEL:$'       "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/base.yaml"
sed 's|^  X_WALL: true$|  X_WALL: false|' "$BASE" > "$TMP/off.yaml"
sed 's|^FLUID DYNAMIC:$|FLUID DYNAMIC:\n  WALL_NORMAL_NODE_DISTANCE: 0.01|' "$BASE" > "$TMP/claimkey.yaml"
sed 's|^FLUID DYNAMIC/WALL MODEL:$|FLUID DYNAMIC/WALL MODEL:\n  XWALL_TAUW_TYPE: "constant"|' "$BASE" > "$TMP/xwallkey.yaml"

# The real key is in the deck and the claimed one is not, before anything runs.
echo "REAL_KEY_IN_UPSTREAM_DECK=$(grep -c '^  X_WALL: ' "$TMP/base.yaml")"
echo "CLAIMED_KEY_IN_UPSTREAM_DECK=$(grep -c 'WALL_NORMAL_NODE_DISTANCE' "$TMP/base.yaml")"
echo "OFF_DECK_HAS_XWALL_ON=$(grep -c '^  X_WALL: true$' "$TMP/off.yaml")"

probe BASE     "$TMP/base.yaml"
probe OFF      "$TMP/off.yaml"
probe CLAIMKEY "$TMP/claimkey.yaml"
probe XWALLKEY "$TMP/xwallkey.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BASE.log")"
echo "BASE_RESULT_PASSES=$(grep -c 'is CORRECT' "$TMP/BASE.log")"

# Turning the real switch off does not degrade the profile, it stops the run.
grep -m1 -F "Teuchos::Exceptions::InvalidParameterName" "$TMP/OFF.log"
grep -m1 -F 'The parameter "xwalltoggle" does not exist' "$TMP/OFF.log"
echo "OFF_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/OFF.log")"
echo "CLAIMED_LOGLAW_TEXT=$(cat "$TMP"/BASE.log "$TMP"/OFF.log | grep -ciE 'log-law|law of the wall|law-of-the-wall')"

# The two key spellings the entry gives are not keys.
grep -m1 -F "Could not match this input" "$TMP/CLAIMKEY.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/CLAIMKEY.log"
echo "CLAIMED_KEY_UNUSED=$(grep -c 'WALL_NORMAL_NODE_DISTANCE: 0.01' "$TMP/CLAIMKEY.log")"
echo "XWALL_PREFIXED_KEY_UNUSED=$(grep -c 'XWALL_TAUW_TYPE: \"constant\"' "$TMP/XWALLKEY.log")"
exit 0
