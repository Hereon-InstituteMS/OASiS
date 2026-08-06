#!/bin/bash
# Tier-2 for fourc::input_format#11 — monolithic FSI matches the two sides of
# the interface by COORDINATE, so the structure and the fluid each need their
# OWN nodes there.  The upstream deck this runs on is built exactly that way:
# nodes 13-16 (fluid) and 17-20 (structure) sit on four identical coordinates,
# a duplication that is deliberate and is what the coupling operator pairs up.
#
# Two failure shapes are pinned, and neither uses the wording the entry quoted:
#
#   one side declared  -> "got 0 master nodes but 4 slave nodes for coupling"
#                         (coupling/src/adapter/4C_coupling_adapter.cpp)
#   neither declared   -> "No nodes in matching FSI interface. Empty FSI
#                          coupling condition?"  (fsi/src/monolithic/…)
#
# The claimed 'no FSI interface nodes found' does not exist, and the claimed
# alternative -- that it "runs without coupling" -- does not happen either: both
# arms abort with exit 1 rather than quietly producing an uncoupled answer.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q 'COUPALGO: "iter_monolithicfluidsplit"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '^DESIGN FSI COUPLING SURF CONDITIONS:$' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/good.yaml"
python3 - "$BASE" "$TMP" <<'PY'
import re, sys
base, tmp = sys.argv[1], sys.argv[2]
b = open(base).read()
both = '''DESIGN FSI COUPLING SURF CONDITIONS:
  - E: 1
    coupling_id: 1
  - E: 2
    coupling_id: 1
'''
fluid_only = '''DESIGN FSI COUPLING SURF CONDITIONS:
  - E: 1
    coupling_id: 1
'''
assert both in b, "upstream FSI coupling block changed"
open(tmp + "/oneside.yaml", "w").write(b.replace(both, fluid_only))
open(tmp + "/nointerface.yaml", "w").write(b.replace(both, ""))

# How many pairs of nodes share a coordinate in the upstream mesh?
co = {}
for m in re.finditer(r'"NODE (\d+) COORD ([^"]+)"', b):
    co.setdefault(m.group(2).strip(), []).append(int(m.group(1)))
dup = [v for v in co.values() if len(v) > 1]
print("UPSTREAM_COINCIDENT_NODE_PAIRS=%d" % len(dup))
print("UPSTREAM_DUPLICATED_NODES=%s" % sorted(n for v in dup for n in v))
PY

probe GOOD        "$TMP/good.yaml"
probe ONESIDE     "$TMP/oneside.yaml"
probe NOINTERFACE "$TMP/nointerface.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
echo "GOOD_CORRECT=$(grep -c 'is CORRECT' "$TMP/GOOD.log")"
grep -m1 -F "got 0 master nodes but 4 slave nodes for coupling" "$TMP/ONESIDE.log"
grep -m1 -F "4C_coupling_adapter.cpp" "$TMP/ONESIDE.log"
grep -m1 -F "No nodes in matching FSI interface. Empty FSI coupling condition?" "$TMP/NOINTERFACE.log"
grep -m1 -F "4C_fsi_monolithic.cpp" "$TMP/NOINTERFACE.log"
# Neither degenerate interface is tolerated: no uncoupled run, no result tests.
echo "ONESIDE_RAN_RESULT_TESTS=$(grep -c 'is CORRECT' "$TMP/ONESIDE.log")"
echo "NOINTERFACE_RAN_RESULT_TESTS=$(grep -c 'is CORRECT' "$TMP/NOINTERFACE.log")"
# The wording the entry quoted does not appear anywhere.
echo "CLAIMED_NO_FSI_INTERFACE_NODES_FOUND_TEXT=$(cat "$TMP/ONESIDE.log" "$TMP/NOINTERFACE.log" | grep -ci 'no FSI interface nodes found')"
exit 0
