#!/bin/bash
# Tier-2: 4C section name must be exactly 'SCALAR TRANSPORT DYNAMIC'.
#
# A common confusion: agents abbreviate the section name to
# 'SCATRA DYNAMIC' (matching the internal application name).
# 4C rejects this with a clear 'is not a valid section name'
# diagnostic.
set -u
BIN=$HOME/Schreibtisch/4C-src/4C/build/4C
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/probe.yaml" <<'EOF'
PROBLEM TYPE:
  PROBLEMTYPE: Scalar_Transport
SCATRA DYNAMIC:
  TIMEINTEGR: Stationary
EOF
"$BIN" "$TMP/probe.yaml" "$TMP/out" 2>&1 | head -20
exit 0
