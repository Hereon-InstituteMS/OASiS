#!/bin/bash
# Tier-2: 4C rejects an invalid PROBLEMTYPE value.
#
# The agent might compose YAML with a misspelled or non-existent
# problem type (e.g. PROBLEMTYPE: Hyperelasticity rather than
# PROBLEMTYPE: Structure). 4C catches this at parse time and
# raises an explicit "Could not match this input" error from
# the input-spec builder, including the offending YAML block.
set -u
BIN=$HOME/Schreibtisch/4C-src/4C/build/4C
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/probe.yaml" <<'EOF'
PROBLEM TYPE:
  PROBLEMTYPE: TotallyMadeUpProblem
  RESTART: 0
EOF
"$BIN" "$TMP/probe.yaml" "$TMP/out" 2>&1 | head -25
exit 0
