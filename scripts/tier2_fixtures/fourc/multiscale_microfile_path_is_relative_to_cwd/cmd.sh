#!/bin/bash
# Tier-2 for fourc::multiscale#1 — what a bad MICROFILE really does, and the
# resolution rule nobody documents.
#
# Claimed: 4C aborts with `failed to load micro input file X` or `micro input
#          file missing MATERIALS section`. Neither string exists in 4C.
# Observed, on upstream sohex8_multiscale_macro:
#   * a MICROFILE that is not there gives the ordinary input-file diagnostic,
#     "Input file 'X' does not exist." from core/io/src/4C_io_input_file.cpp —
#     nothing mentions micro, multiscale or homogenisation, so an agent grepping
#     for those words finds nothing.
#   * the path is resolved relative to the WORKING DIRECTORY, not to the macro
#     deck. The identical deck runs from one directory and fails from another,
#     which is the trap: the upstream regression suite runs with cwd set to the
#     deck directory, so copying the macro deck somewhere else silently breaks it.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sohex8_multiscale_macro.4C.yaml) || exit 3
MICRO=$(upstream sohex8_multiscale_micro.mat.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" macro.yaml
cp "$MICRO" .
grep -q 'MICROFILE: "sohex8_multiscale_micro.mat.4C.yaml"' macro.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# Arm 1: micro file beside the deck AND in the working directory -> runs.
probe BESIDE macro.yaml

# Arm 2: identical deck, run from a subdirectory. The micro file is no longer in
# the working directory, so the same input fails.
mkdir -p sub && cp macro.yaml sub/
( cd sub && stdbuf -oL -eL "$BIN" macro.yaml res > run.log 2>&1; echo "EXIT_FROM_SUBDIR=$?" )

# Arm 3: micro file genuinely absent.
sed 's/MICROFILE: "sohex8_multiscale_micro.mat.4C.yaml"/MICROFILE: "no_such_micro_file.4C.yaml"/' \
    macro.yaml > missing.yaml
probe MISSING missing.yaml

echo "BESIDE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BESIDE.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BESIDE.log"
grep -m1 -F "Input file 'no_such_micro_file.4C.yaml' does not exist." "$TMP/MISSING.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/MISSING.log"
grep -m1 -F "Input file 'sohex8_multiscale_micro.mat.4C.yaml' does not exist." sub/run.log
# The diagnostic never mentions the micro scale at all.
echo "MISSING_MENTIONS_MICRO=$(grep -m1 -A1 'does not exist' "$TMP/MISSING.log" | grep -ci 'micro input\|multiscale\|homogeni')"
echo "CLAIMED_FAILED_TO_LOAD_TEXT=$(grep -ci 'failed to load micro input file' "$TMP/MISSING.log")"
exit 0
