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
BIN=$HOME/Schreibtisch/4C-src/4C/build/4C
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/probe.dat" <<'EOF'
-------------------------------------------------------------------TITLE
Probe — dat-style input on YAML-only binary
------------------------------------------------------------PROBLEM TYPE
PROBLEMTYP                      Structure
RESTART                         0
EOF

# Pipe output for the expect_in_output check.
"$BIN" "$TMP/probe.dat" "$TMP/out" 2>&1 | head -30
exit 0
