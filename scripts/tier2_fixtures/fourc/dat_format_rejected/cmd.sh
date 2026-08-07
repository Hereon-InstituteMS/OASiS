#!/bin/bash
# Tier-2: 4C 2026.3.0-dev rejects .dat input format.
#
# Pitfall (4C input format): The current 4C binary only reads
# .yaml / .yml / .json. The old dat-format with section headers
# like "------TITLE" is rejected at file-read time with a clear
# diagnostic from core/io/src/4C_io_input_file.cpp:428.
#
# The fixture writes a minimal dat-style input, invokes 4C, and
# expects the specific rejection text.

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

# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology — the same content is
# offered to 4C as a .yaml file with YAML syntax, i.e. the format the binary
# does support.  4C then infers the format, gets past the file-read stage and
# never prints "Cannot infer format of input file", so the fixture must go red.
MUTATE="${T2_MUTATE:-0}"

if [ "$MUTATE" = "1" ]; then
  PROBE="$TMP/probe.yaml"
  cat > "$PROBE" <<'EOF'
TITLE:
  - "Probe — YAML input on YAML-only binary"
PROBLEM TYPE:
  PROBLEMTYPE: Structure
  RESTART: 0
EOF
else
  PROBE="$TMP/probe.dat"
  cat > "$PROBE" <<'EOF'
-------------------------------------------------------------------TITLE
Probe — dat-style input on YAML-only binary
------------------------------------------------------------PROBLEM TYPE
PROBLEMTYP                      Structure
RESTART                         0
EOF
fi

# Pipe output for the expect_in_output check.
stdbuf -oL -eL "$BIN" "$PROBE" "$TMP/out" 2>&1 | head -30
exit 0
