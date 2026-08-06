#!/bin/bash
# Tier-2 for fourc::beams#8 — a DESIGN POINT DIRICH/NEUMANN condition needs a
# DNODE-NODE TOPOLOGY block that maps design node IDs to mesh node IDs. Delete
# the block and the run dies. But the message is not the one the entry quoted.
#
# Claimed:  'no design nodes found' from 4C_io_input_file.cpp.
# Observed: neither that text nor that file. What 4C prints is
#
#     DPoint 1 not in range [0:0[
#     DPoint condition on non existent DPoint?Could not read set from entity type.
#     .../core/fem/src/condition/4C_fem_condition.cpp
#
#   The "[0:0[" is the useful part — it is the size of the design-point list,
#   i.e. it says the list is empty — and the "1" is the ZERO-BASED index, so a
#   condition written as "E: 2" reports DPoint 1. Both are easy to misread.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_static_test1.4C.yaml) || exit 3

python3 - "$BASE" "$TMP/no_topology.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = '''DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
  - "NODE 6 DNODE 2"
'''
assert blk in t, "upstream deck no longer carries the two-entry DNODE-NODE TOPOLOGY"
open(sys.argv[2], "w").write(t.replace(blk, "", 1))
PY

cp "$BASE" "$TMP/with_topology.yaml"

probe WITHTOPO "$TMP/with_topology.yaml"
probe NOTOPO   "$TMP/no_topology.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHTOPO.log"
grep -m1 -F "DPoint 1 not in range [0:0[" "$TMP/NOTOPO.log"
grep -m1 -F "DPoint condition on non existent DPoint?Could not read set from entity type." "$TMP/NOTOPO.log"
grep -m1 -F "4C_fem_condition.cpp" "$TMP/NOTOPO.log"
# The quoted diagnostic and the quoted source file are both absent.
echo "CLAIMED_NO_DESIGN_NODES_TEXT=$(grep -ci 'no design nodes found' "$TMP/NOTOPO.log")"
echo "CLAIMED_SOURCE_FILE=$(grep -c '4C_io_input_file.cpp' "$TMP/NOTOPO.log")"
exit 0
