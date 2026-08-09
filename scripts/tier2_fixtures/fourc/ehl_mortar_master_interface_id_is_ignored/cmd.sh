#!/bin/bash
# Tier-2 for fourc::ehl#0 — the EHL lubrication/structure interface is NOT
# checked for consistency, and the failure is silent in both directions.
#
# Claimed:  "mismatched node positions on the contact face raise 'no matching
#            lubrication interface' from 4C_ehl_factory.cpp at setup".
# Observed, on upstream ehl3d_mixed.4C.yaml (DESIGN SURF EHL MORTAR COUPLING
# CONDITIONS 3D, Slave on E:2 with InterfaceID 1, Master on E:3 with
# InterfaceID 1, seven result tests):
#
#   ID_MISMATCH : give the Master InterfaceID 2 while the Slave keeps 1 — i.e.
#                 declare two different interfaces.  4C accepts it, runs, and
#                 reproduces all seven reference values to 1e-16.  The
#                 InterfaceID is simply not used to pair the sides here.  Zero
#                 diagnostics.
#   NO_MASTER   : delete the Master condition entirely.  Still no abort and no
#                 message naming the interface: the run completes, and the
#                 structure just follows its own Dirichlet with the contact
#                 node displacement collapsing to ~1e-15 in x and exactly the
#                 prescribed -5e-2 in y.  Only the deck's own result test
#                 notices, 5 of 7.
#
# There is no 4C_ehl_factory.cpp in 4C and no 'no matching lubrication
# interface' anywhere.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ehl3d_mixed.4C.yaml) || exit 3
grep -q '^DESIGN SURF EHL MORTAR COUPLING CONDITIONS 3D:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_ehl_coupling_section_changed"; exit 3; }
grep -q '    Side: "Master"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_master_side_changed"; exit 3; }

# The two pathologies.
MASTER_INTERFACE_ID=2
DROP_MASTER=yes

cp "$BASE" "$TMP/matched.yaml"
python3 - "$BASE" "$TMP/idmismatch.yaml" "$MASTER_INTERFACE_ID" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = '  - E: 3\n    InterfaceID: 1\n    Side: "Master"'
assert old in t, "upstream deck no longer declares the Master side on E:3/ID 1"
open(sys.argv[2], "w").write(
    t.replace(old, '  - E: 3\n    InterfaceID: %s\n    Side: "Master"' % sys.argv[3]))
PY
python3 - "$BASE" "$TMP/nomaster.yaml" "$DROP_MASTER" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = ('  - E: 3\n    InterfaceID: 1\n    Side: "Master"\n'
       '    Initialization: "Inactive"\n    FrCoeffOrBound: 0.3\n')
assert blk in t, "upstream deck no longer carries the Master coupling block"
if sys.argv[3] == "yes":
    t = t.replace(blk, "")
open(sys.argv[2], "w").write(t)
PY
grep -c 'Side: "Master"' "$TMP/nomaster.yaml" | sed 's/^/MASTER_BLOCKS_LEFT=/'
# Record that the mismatch really was written into the deck, so that reverting
# it cannot leave the fixture green on a deck that is no longer mismatched.
python3 - "$TMP/idmismatch.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
ids = re.findall(r'InterfaceID: (\d+)\n    Side: "(\w+)"', t)
d = dict((side, i) for i, side in ids)
print("SLAVE_MASTER_IDS_DIFFER=%s"
      % ("yes" if d.get("Slave") != d.get("Master") else "no"))
PY

probe MATCHED    "$TMP/matched.yaml"
probe IDMISMATCH "$TMP/idmismatch.yaml"
probe NOMASTER   "$TMP/nomaster.yaml"

grep -m1 -F "OK (7)" "$TMP/MATCHED.log"
grep -m1 -F "OK (7)" "$TMP/IDMISMATCH.log"
grep -m1 -F "processor 0 finished normally" "$TMP/IDMISMATCH.log"
grep -m1 -F "Result check failed with 5 errors out of 7 tests" "$TMP/NOMASTER.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NOMASTER.log"

echo "IDMISMATCH_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/IDMISMATCH.log")"
echo "IDMISMATCH_PASSED_TESTS=$(grep -c 'is CORRECT' "$TMP/IDMISMATCH.log")"
echo "NOMASTER_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/NOMASTER.log")"
# Not one word about the interface, in either broken arm.
echo "CLAIMED_NO_MATCHING_TEXT=$(grep -ci 'no matching lubrication interface' "$TMP/IDMISMATCH.log")$(grep -ci 'no matching lubrication interface' "$TMP/NOMASTER.log")"
echo "CLAIMED_EHL_FACTORY_FILE=$(grep -c '4C_ehl_factory' "$TMP/NOMASTER.log")"
echo "INTERFACE_DIAGNOSTICS=$(grep -ciE 'interface.*(mismatch|not found|missing|inconsistent)|unpaired' "$TMP/IDMISMATCH.log")"
exit 0
