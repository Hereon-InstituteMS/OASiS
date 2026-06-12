#!/bin/bash
# Tier-2: STRUCTURAL DYNAMIC.DYNAMICTYPE enum validation.
#
# Inside the STRUCTURAL DYNAMIC section, DYNAMICTYPE must be
# one of the enum values 4C recognises (Statics, GenAlpha,
# OneStepTheta, etc.). A made-up value is rejected at parse
# time with the section block echoed.
set -u
BIN=$HOME/Schreibtisch/4C-src/4C/build/4C
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/probe.yaml" <<'EOF'
PROBLEM TYPE:
  PROBLEMTYPE: Structure
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: TotallyMadeUpScheme
EOF
"$BIN" "$TMP/probe.yaml" "$TMP/out" 2>&1 | head -25
exit 0
