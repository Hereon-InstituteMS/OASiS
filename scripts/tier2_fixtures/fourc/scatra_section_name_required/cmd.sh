#!/bin/bash
# Tier-2: 4C section name must be exactly 'SCALAR TRANSPORT DYNAMIC'.
#
# A common confusion: agents abbreviate the section name to
# 'SCATRA DYNAMIC' (matching the internal application name).
# 4C rejects this with a clear 'is not a valid section name'
# diagnostic.
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
cat > "$TMP/probe.yaml" <<'EOF'
PROBLEM TYPE:
  PROBLEMTYPE: Scalar_Transport
SCATRA DYNAMIC:
  TIMEINTEGR: Stationary
EOF
stdbuf -oL -eL "$BIN" "$TMP/probe.yaml" "$TMP/out" 2>&1 | head -20
exit 0
