#!/bin/bash
# Tier-2 for fourc::fbi#2 — a beam that lies outside the fluid mesh does NOT
# raise an error.  It just never feels the fluid.
#
# Claimed:  "an open fluid domain that ends at the beam region produces a
#            'beam outside fluid domain' error from 4C_fbi_partitioner.cpp".
# Observed: upstream fbi_mortar_solidcoupling.4C.yaml has a fluid cube
#           [-0.5,0.5]^3 and one BEAM3R inside it.  Translate the beam to x = 5,
#           entirely outside the fluid, and 4C:
#             * builds both fields and every beam-fluid search structure,
#             * prints no 'outside', no 'not found', no search diagnostic,
#             * reports NOX "The solution passed into the solver ... is already
#               converged!  The solver wil not attempt to solve this system"
#               (4C's own spelling) — because the beam carries no fluid load,
#             * ends with the beam displacement EXACTLY 0.0 in x.
#           The only failure is the deck's own result test noticing.
#           Half-outside is worse: 4 of the 6 tests move and still nothing warns.
#
# So the hazard is silent decoupling, not a partitioner error, and there is no
# 4C_fbi_partitioner.cpp in 4C at all.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fbi_mortar_solidcoupling.4C.yaml) || exit 3
grep -q '"NODE 1 COORD 0.0 -0.5 0"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_beam_nodes_changed"; exit 3; }
grep -q 'top_corner_point: \[0.5, 0.5, 0.5\]' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fluid_domain_changed"; exit 3; }
cp "$(dirname "$BASE")/beam_flow_solver.xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_nox_xml"; exit 3; }
cd "$TMP" || exit 3

# The pathology: how far the beam is pushed out of the fluid cube in +x.
BEAM_X_OFFSET=5.0

cp "$BASE" "$TMP/inside.yaml"
python3 - "$BASE" "$TMP/outside.yaml" "$BEAM_X_OFFSET" <<'PY'
import re, sys
t, off = open(sys.argv[1]).read(), float(sys.argv[3])
n = 0
def shift(m):
    global n
    n += 1
    return '"NODE %s COORD %g %s %s"' % (m.group(1), float(m.group(2)) + off,
                                         m.group(3), m.group(4))
t = re.sub(r'"NODE (\d+) COORD ([-0-9.eE+]+) ([-0-9.eE+]+) ([-0-9.eE+]+)"', shift, t)
assert n == 3, "expected exactly the 3 explicit beam nodes, found %d" % n
open(sys.argv[2], "w").write(t)
PY
# Half-outside variant: same offset applied to only part of the beam.
python3 - "$BASE" "$TMP/halfout.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
for a, b in ((' 0.0 -0.5 0"', ' 0.0 0.0 0"'),
             (' 0.0 0.0 0"', ' 0.0 1.0 0"'),
             (' 0.0 -0.25 0.0"', ' 0.0 0.5 0.0"')):
    assert a in t, a
    t = t.replace(a, b, 1)
open(sys.argv[2], "w").write(t)
PY

probe INSIDE  "$TMP/inside.yaml"
probe OUTSIDE "$TMP/outside.yaml"
probe HALFOUT "$TMP/halfout.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/INSIDE.log"
grep -m1 -F "OK (6)" "$TMP/INSIDE.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/OUTSIDE.log"
grep -m1 -F "The solver wil not attempt to solve this system" "$TMP/OUTSIDE.log"
grep -m1 -F "Result check failed with 2 errors out of 6 tests" "$TMP/OUTSIDE.log"
grep -m1 -F "Result check failed with 4 errors out of 6 tests" "$TMP/HALFOUT.log"

# Nothing at all is said about the beam leaving the fluid.
echo "CLAIMED_BEAM_OUTSIDE_TEXT=$(grep -ci 'beam outside fluid domain' "$TMP/OUTSIDE.log")"
echo "CLAIMED_FBI_PARTITIONER_FILE=$(grep -c '4C_fbi_partitioner' "$TMP/OUTSIDE.log")"
echo "OUTSIDE_SEARCH_DIAGNOSTICS=$(grep -ciE 'outside (the )?fluid|no host element|gauss point.*(not|fail).*project|element not found' "$TMP/OUTSIDE.log")"
# Both fields were still built and the run completed its time loop.
echo "OUTSIDE_FIELDS_BUILT=$(grep -c 'fill_complete() on discretization' "$TMP/OUTSIDE.log")"
# The beam simply did not move.
if grep -qE "dispx +at node +1.*actresult= 0\.00000000000000000e\+00" "$TMP/OUTSIDE.log" \
   && grep -qE "dispx +at node +2.*actresult= 0\.00000000000000000e\+00" "$TMP/OUTSIDE.log"; then
  echo "OUTSIDE_BEAM_DISPLACEMENT=exactly_zero"
else
  echo "OUTSIDE_BEAM_DISPLACEMENT=nonzero"
fi
exit 0
