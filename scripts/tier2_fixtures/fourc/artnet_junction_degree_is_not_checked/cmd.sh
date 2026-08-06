#!/bin/bash
# Tier-2 for fourc::arterial_network#3 — junction DEGREE is never checked; what
# 4C does check is that a junction has at least two nodes and is not all-outlet.
#
# Claimed: "a mis-connected junction node (4 segments when 3 expected, or shared
#          node IDs across non-adjacent segments) raises 'inconsistent junction
#          connectivity' at network setup."
# Observed, on upstream one_d_3_artery_network shortened to 50 steps, whose
# junction 1 joins three branch nodes:
#   * adding a FOURTH node to the same junction id is accepted without a word.
#     The run completes its 50 steps and reaches the result test; only the numbers
#     change. There is no expected degree.
#   * the two real junction diagnostics live in art_net/4C_art_net_art_junction.cpp
#     and are about something else entirely:
#       - a junction with a single member -> "An arterial junction is supposed to
#         have at least two nodes!"
#       - every member flagged as an outlet -> "Junction (1) has all of its nodes
#         defined as outlets"
#   * "inconsistent junction connectivity" appears nowhere.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream one_d_3_artery_network.4C.yaml) || exit 3
cd "$TMP" || exit 3
grep -q '  NUMSTEP: 10000' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
sed 's/  NUMSTEP: 10000/  NUMSTEP: 50/' "$BASE" > base.yaml

python3 - <<'PY'
t = open('base.yaml').read()
J = '''DESIGN NODE 1D ARTERY JUNCTION CONDITIONS:
  - E: 2
    ConditionID: 1
    Kr: 0.1
  - E: 3
    ConditionID: 1
    Kr: 0.2
  - E: 5
    ConditionID: 1
    Kr: 0.3
'''
assert J in t, "upstream junction condition changed"
IO = '''DESIGN NODE 1D ARTERY IN_OUTLET CONDITIONS:
  - E: 1
  - E: 2
    terminaltype: "outlet"
  - E: 3
  - E: 4
    terminaltype: "outlet"
  - E: 5
  - E: 6
    terminaltype: "outlet"
'''
assert IO in t, "upstream in/outlet condition changed"

# a fourth branch on the same junction id
FOURTH_BRANCH = '  - E: 4\n    ConditionID: 1\n    Kr: 0.4\n'
# a junction with a single member
SINGLETON_JUNCTION = 'DESIGN NODE 1D ARTERY JUNCTION CONDITIONS:\n  - E: 2\n    ConditionID: 1\n    Kr: 0.1\n'
# every junction member declared an outlet
ALLOUT_INOUT = 'DESIGN NODE 1D ARTERY IN_OUTLET CONDITIONS:\n  - E: 1\n  - E: 2\n    terminaltype: "outlet"\n  - E: 3\n    terminaltype: "outlet"\n  - E: 4\n    terminaltype: "outlet"\n  - E: 5\n    terminaltype: "outlet"\n  - E: 6\n    terminaltype: "outlet"\n'

open('fourway.yaml', 'w').write(t.replace(J, J + FOURTH_BRANCH))
open('singleton.yaml', 'w').write(t.replace(J, SINGLETON_JUNCTION))
open('allout.yaml', 'w').write(t.replace(IO, ALLOUT_INOUT))
PY

probe BASE      base.yaml
probe FOURWAY   fourway.yaml
probe SINGLETON singleton.yaml
probe ALLOUT    allout.yaml

echo "FOURWAY_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/FOURWAY.log")"
echo "SINGLETON_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SINGLETON.log")"
echo "ALLOUT_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/ALLOUT.log")"
grep -m1 -F "area     at node   1" "$TMP/BASE.log"
grep -m1 -F "area     at node   1" "$TMP/FOURWAY.log"
grep -m1 -F "An arterial junction is supposed to have at least two nodes!" "$TMP/SINGLETON.log"
grep -m1 -F "Junction (1) has all of its nodes defined as outlets" "$TMP/ALLOUT.log"
grep -m1 -F "4C_art_net_art_junction.cpp" "$TMP/SINGLETON.log"
python3 - "$TMP" <<'PY'
import re, sys
def val(p):
    for line in open(p):
        m = re.search(r'area     at node   1\s+is WRONG --> actresult=\s*(\S+),', line)
        if m:
            return float(m.group(1))
    raise SystemExit("no area line in " + p)
b, f = val(sys.argv[1] + '/BASE.log'), val(sys.argv[1] + '/FOURWAY.log')
print("FOURWAY_CHANGED_THE_ANSWER=%s" % ("yes" if b != f else "no"))
PY
echo "FOURWAY_DEGREE_DIAGNOSTICS=$(grep -ciE 'has degree|wrong degree|expected number of|too many (branches|segments|nodes)|mis-connected|inconsistent' "$TMP/FOURWAY.log")"
echo "CLAIMED_INCONSISTENT_TEXT=$(cat "$TMP"/FOURWAY.log "$TMP"/SINGLETON.log "$TMP"/ALLOUT.log | grep -ci 'inconsistent junction connectivity')"
exit 0
