#!/bin/bash
# Tier-2 for fourc::cardiovascular0d#2 — an open cavity boundary gives a wrong
# volume, and 4C shows you the wrong number every step before it gives up.
#
# Claim: "cavity volume computed from SURFACE INTEGRAL over the closed cavity
#        boundary ... an OPEN cavity boundary (mesh hole or missing surface) gives
#        wrong volume -- the surface integral is incorrect and the 0D-3D coupling
#        drifts."
# Observed, on upstream cardiovascular0d_4elementwindkessel_structure_direct_
# genalpha, whose cavity 0 is DSURFACE 3 = all eight nodes of the first HEX8, i.e.
# the complete closed boundary of a 10x10x10 cube:
#   * closed: 4C prints "0 V:" near -1.0e+03 each step, drifting with the load,
#     and the run exits 0 with all three result tests correct.
#   * drop nodes 5..8 from DSURFACE 3, leaving one face: the printed volume becomes
#     -1.6666666666666666e+02 and never moves. That is exactly (1/3)*(-5)*100, the
#     divergence-theorem integral over the single remaining face -- a sixth of the
#     truth, and constant, so the 0D model sees a rigid cavity.
#   * the run does not silently continue: NOX gives up with "The nonlinear solver
#     did not converge!" and no result test runs. But nothing in that message
#     mentions the cavity surface, so the -166.67 in the output is the only clue.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream cardiovascular0d_4elementwindkessel_structure_direct_genalpha.4C.yaml) || exit 3
NOXXML=$(upstream cardiovascular0d_new_struc.xml) || exit 3
cd "$TMP" || exit 3
cp "$NOXXML" .
cp "$BASE" base.yaml
for n in 5 6 7 8; do
  grep -q "\"NODE $n DSURFACE 3\"" base.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
done

OPEN_THE_CAVITY=yes

python3 - "$OPEN_THE_CAVITY" <<'PY'
import sys
t = open('base.yaml').read()
o = t
if sys.argv[1] == 'yes':
    for n in (5, 6, 7, 8):
        o = o.replace('  - "NODE %d DSURFACE 3"\n' % n, '')
    assert o != t
open('open.yaml', 'w').write(o)
PY

probe CLOSED base.yaml
probe OPEN   open.yaml

echo "CLOSED_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/CLOSED.log")"
echo "OPEN_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/OPEN.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/CLOSED.log"
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/OPEN.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/OPEN.log"
echo "CLOSED_CAVITY0_VOLUMES=$(grep -A2 'Cardiovascular0D output id 0' "$TMP/CLOSED.log" | grep -c ' 0 V:')"
grep -A2 'Cardiovascular0D output id 0' "$TMP/CLOSED.log" | grep -m1 ' 0 V:'
grep -A2 'Cardiovascular0D output id 0' "$TMP/OPEN.log"   | grep -m1 ' 0 V:'
echo "OPEN_DISTINCT_CAVITY0_VOLUMES=$(grep -A2 'Cardiovascular0D output id 0' "$TMP/OPEN.log" | grep ' 0 V:' | sort -u | wc -l)"
echo "OPEN_VOLUME_IS_ONE_FACE_INTEGRAL=$(grep -A2 'Cardiovascular0D output id 0' "$TMP/OPEN.log" | grep -c ' 0 V: -1.6666666666666666e+02')"
echo "OPEN_CAVITY_DIAGNOSTICS=$(grep -ciE 'open (surface|cavity)|not closed|surface integral' "$TMP/OPEN.log")"
exit 0
