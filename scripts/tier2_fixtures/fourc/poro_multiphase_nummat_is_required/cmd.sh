#!/bin/bash
# Tier-2 for fourc::porous_media#3 — NUMMAT in MAT_FluidPoroMultiPhase is a
# REQUIRED key with no default.  Deleting it is a hard abort during
# read_materials, and the message is
#
#     Parameter 'NUMMAT' not found in container.   (global_data/4C_global_data_read.cpp)
#
# not the 'material list size mismatch' the entry quoted — that string is
# nowhere in 4C.  Asserted both ways.
#
# The failure happens while the MATERIALS section is being read, i.e. long
# before any "size mismatch" between a list and its declared length could be
# detected: there is no defaulting step to mismatch against.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream porofluid_pressure_based_2D_quad4.4C.yaml) || exit 3
ln -s "$(dirname "$BASE")/xml" "$TMP/xml"
grep -q "      NUMMAT: 4" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/with.yaml"

python3 - "$BASE" "$TMP/without.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = "      NUMMAT: 4\n      MATIDS: [10, 11, 12, 13]\n"
if blk not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(
    t.replace(blk, "      MATIDS: [10, 11, 12, 13]\n", 1))
PY
[ -f "$TMP/without.yaml" ] || exit 3
# MATIDS still lists four sub-materials — only the count is gone.
grep -q "MATIDS: \[10, 11, 12, 13\]" "$TMP/without.yaml" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

probe WITH    "$TMP/with.yaml"
probe WITHOUT "$TMP/without.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITH.log"
grep -m1 -F "Parameter 'NUMMAT' not found in container." "$TMP/WITHOUT.log"
grep -m1 -oF "4C_global_data_read.cpp" "$TMP/WITHOUT.log"
# It dies inside read_materials, not at a later consistency check.
echo "FAILS_IN_READ_MATERIALS=$(grep -c 'Global::read_materials' "$TMP/WITHOUT.log")"
# The quoted wording does not exist.
echo "CLAIMED_SIZE_MISMATCH_TEXT=$(grep -ci 'material list size mismatch' "$TMP/WITHOUT.log")"
exit 0
