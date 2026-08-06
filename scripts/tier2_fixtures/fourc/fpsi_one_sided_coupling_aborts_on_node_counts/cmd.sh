#!/bin/bash
# Tier-2 for fourc::fpsi#1 — the interface does have to be declared from both
# sides, but a one-sided declaration does NOT "give zero force transfer".  It
# aborts, and the diagnostic is a node-count mismatch.
#
# Claimed:  "defining only the fluid-side FPSI coupling condition gives zero
#            force transfer to the porous skeleton (it does not deform)".
# Observed: upstream fpsi_ofsiinterface.4C.yaml declares
#             DESIGN FPSI COUPLING SURF CONDITIONS: E 7 and E 6, both coupling_id 1.
#           Delete the E 6 entry and 4C aborts before the first time step with
#             "got 4 master nodes but 0 slave nodes for coupling"
#           from coupling/src/adapter/4C_coupling_adapter.cpp line 69.  The run
#           never deforms anything because it never runs.
#
# The message is worth pinning because it names neither FPSI nor the condition:
# a reader has to know that "slave nodes" means the side they forgot to declare.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fpsi_ofsiinterface.4C.yaml) || exit 3
grep -q '^DESIGN FPSI COUPLING SURF CONDITIONS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fpsi_condition_section_changed"; exit 3; }

# The pathology: drop the second side of the FPSI interface.
DROP_SECOND_SIDE=yes

cp "$BASE" "$TMP/bothsides.yaml"
python3 - "$BASE" "$TMP/oneside.yaml" "$DROP_SECOND_SIDE" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = '  - E: 6\n    coupling_id: 1\n'
head = 'DESIGN FPSI COUPLING SURF CONDITIONS:\n  - E: 7\n    coupling_id: 1\n' + blk
assert head in t, "upstream deck no longer declares both FPSI coupling sides"
if sys.argv[3] == "yes":
    t = t.replace(head, head[:-len(blk)], 1)
open(sys.argv[2], "w").write(t)
PY
python3 - "$TMP/oneside.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r'DESIGN FPSI COUPLING SURF CONDITIONS:\n((?:  - E: \d+\n    coupling_id: \d+\n)+)', t)
print("FPSI_COUPLING_SIDES_DECLARED=%d" % m.group(1).count("- E:"))
PY

probe BOTHSIDES "$TMP/bothsides.yaml"
probe ONESIDE   "$TMP/oneside.yaml"

grep -m1 -F "OK (2)" "$TMP/BOTHSIDES.log"
grep -m1 -F "processor 0 finished normally" "$TMP/BOTHSIDES.log"
grep -m1 -F "got 4 master nodes but 0 slave nodes for coupling" "$TMP/ONESIDE.log"
grep -m1 -F "4C_coupling_adapter.cpp" "$TMP/ONESIDE.log"

# It aborts, so there is no "runs but transfers nothing" behaviour to observe.
echo "ONESIDE_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/ONESIDE.log")"
echo "ONESIDE_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/ONESIDE.log")"
# The diagnostic names neither FPSI nor the condition the reader must add.
echo "ONESIDE_MENTIONS_FPSI=$(grep -c 'got 4 master nodes but 0 slave nodes for coupling.*FPSI' "$TMP/ONESIDE.log")"
echo "ONESIDE_MENTIONS_CONDITION_NAME=$(grep -c 'DESIGN FPSI COUPLING SURF CONDITIONS' "$TMP/ONESIDE.log")"
exit 0
