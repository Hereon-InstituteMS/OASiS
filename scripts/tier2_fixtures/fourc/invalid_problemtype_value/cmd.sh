#!/bin/bash
# Tier-2: 4C rejects an invalid PROBLEMTYPE value.
#
# The agent might compose YAML with a misspelled or non-existent
# problem type (e.g. PROBLEMTYPE: Hyperelasticity rather than
# PROBLEMTYPE: Structure). 4C catches this at parse time and
# raises an explicit "Could not match this input" error from
# the input-spec builder, including the offending YAML block.
set -u
# Resolve the 4C binary: explicit override first, then the paths this
# repo has been verified against. Updated 2026-08-03 — the previous
# hard-coded $HOME/Schreibtisch/4C-src path no longer exists on the
# verification host; the deployed build is /home/alexander/4C/build/4C.
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology: the deck names a REAL
# problem type, so the input spec matches, no "Could not match this input" block
# is printed, and the fixture must go red.
MUTATE="${T2_MUTATE:-0}"
PT="TotallyMadeUpProblem"
[ "$MUTATE" = "1" ] && PT="Structure"

cat > "$TMP/probe.yaml" <<EOF
PROBLEM TYPE:
  PROBLEMTYPE: $PT
  RESTART: 0
EOF
stdbuf -oL -eL "$BIN" "$TMP/probe.yaml" "$TMP/out" 2>&1 | head -25
exit 0
