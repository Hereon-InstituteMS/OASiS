#!/bin/bash
# Tier-2: STRUCTURAL DYNAMIC.DYNAMICTYPE enum validation.
#
# Inside the STRUCTURAL DYNAMIC section, DYNAMICTYPE must be
# one of the enum values 4C recognises (Statics, GenAlpha,
# OneStepTheta, etc.). A made-up value is rejected at parse
# time with the section block echoed.
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
# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology: DYNAMICTYPE carries a
# value that IS in the enum, the section matches its spec, no "Could not match
# this input" block is echoed and the fixture must go red.
MUTATE="${T2_MUTATE:-0}"
DT="TotallyMadeUpScheme"
[ "$MUTATE" = "1" ] && DT="Statics"

cat > "$TMP/probe.yaml" <<EOF
PROBLEM TYPE:
  PROBLEMTYPE: Structure
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: $DT
EOF
stdbuf -oL -eL "$BIN" "$TMP/probe.yaml" "$TMP/out" 2>&1 | head -25
exit 0
