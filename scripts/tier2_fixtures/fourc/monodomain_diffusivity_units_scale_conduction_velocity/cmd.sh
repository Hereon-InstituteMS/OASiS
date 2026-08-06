#!/bin/bash
# Tier-2 for fourc::cardiac_monodomain#4 — a diffusivity given in the wrong length
# unit is accepted without comment and rescales the conduction velocity.
#
# Claim: "Cardiac EP units are typically mm, ms, mV, uA/cm^2 ... mixing in SI
#        (m, s, V, A/m^2) produces silent unit conversion errors -- conduction
#        velocity off by 10x or 100x."
# The upstream Cardiac_Monodomain decks are single elements, where the diffusion
# tensor does nothing at all, so the fixture builds a propagating wave: a HEX8
# strip with the same MAT_myocard FHN material as scatra_myocard_FHN_material.
# Front positions come from 4C's own result-test "actresult=" lines.
# Observed: writing DIFF in m^2 instead of mm^2 -- the 1e-6 factor between the two
# -- does not merely slow the wave by 10x or 100x, it stops it. With the correct
# DIFF the phi=0.5 front is at 0.19743 after t=80, i.e. 0.16543 beyond the edge of
# the 0.032-long excited region. With the SI value the front ends at 0.01972, which
# is BEHIND where it started: the excited patch decays in place and nothing
# propagates. The run still converges and exits through the result test, and
# UNIT_WARNINGS=0 -- 4C has no idea which length unit you meant.
. "$(dirname "$0")/../_lib/preamble.sh"

FHN=$(upstream scatra_myocard_FHN_material.4C.yaml) || exit 3
grep -q "      DIFF1: 0.0001171" "$FHN" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "      MODEL: \"FHN\"" "$FHN"     || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cd "$TMP" || exit 3

cat > gen.py <<'PYGEN'
import sys, json
nx = int(sys.argv[1]); L = 0.4; T = 80.0; dt = 0.1
d1, d2, d3, fib, out = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
h = L / nx; w = h
def N(i, c): return i * 4 + c + 1
nodes, k = [], 0
for i in range(nx + 1):
    for (y, z) in [(0, w), (0, 0), (w, 0), (w, w)]:
        k += 1
        nodes.append('  - "NODE %d COORD %.10e %.10e %.10e"' % (k, i * h, y, z))
els = ['  - "%d TRANSP HEX8 %d %d %d %d %d %d %d %d MAT 1 TYPE CardMono FIBER1 %s"'
       % (e + 1, N(e, 0), N(e, 1), N(e, 2), N(e, 3),
          N(e + 1, 0), N(e + 1, 1), N(e + 1, 2), N(e + 1, 3), fib) for e in range(nx)]
probe = [(N(i, 0), i * h) for i in range(nx + 1)]
rd = ''.join('  - SCATRA:\n      DIS: "scatra"\n      NODE: %d\n      QUANTITY: "phi"\n'
             '      VALUE: -98765.0\n      TOLERANCE: 1e-12\n' % n for n, _ in probe)
open(out, 'w').write('''TITLE:
  - "FHN monodomain strip"
PROBLEM TYPE:
  PROBLEMTYPE: "Cardiac_Monodomain"
IO:
  STRUCT_DISP: false
SCALAR TRANSPORT DYNAMIC:
  SOLVERTYPE: "nonlinear"
  MAXTIME: %s
  TIMESTEP: %s
  NUMSTEP: 1000000
  RESULTSEVERY: 1000000
  RESTARTEVERY: 1000000
  MATID: 1
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  SKIPINITDER: true
  LINEAR_SOLVER: 1
CARDIAC MONODOMAIN CONTROL:
  WRITEMAXINTSTATE: 1
  WRITEMAXIONICCURRENTS: 2
SCALAR TRANSPORT DYNAMIC/NONLINEAR:
  CONVTOL: 1e-05
  ABSTOLRES: 1e-10
SCALAR TRANSPORT DYNAMIC/STABILIZATION:
  STABTYPE: "no_stabilization"
  DEFINITION_TAU: "Zero"
  EVALUATION_TAU: "integration_point"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Sca_Tra_Solver"
MATERIALS:
  - MAT: 1
    MAT_myocard:
      DIFF1: %s
      DIFF2: %s
      DIFF3: %s
      PERTURBATION_DERIV: 1e-06
      MODEL: "FHN"
      TIME_SCALE: 1
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0/(1.0+exp((x-%.10e)/%.10e))"
NODE COORDS:
%s
TRANSPORT ELEMENTS:
%s
RESULT DESCRIPTION:
%s''' % (T, dt, d1, d2, d3, 0.08 * L, 0.005 * L, '\n'.join(nodes), '\n'.join(els), rd))
json.dump({str(n): x for n, x in probe}, open(out + '.nodes', 'w'))
PYGEN

cat > front.py <<'PYF'
import re, json, sys
m = json.load(open(sys.argv[1] + '.nodes'))
pts = sorted((m[g.group(1)], float(g.group(2))) for g in
             (re.search(r'at node\s+(\d+)\s+is WRONG --> actresult=\s*(\S+),', line)
              for line in open(sys.argv[2])) if g)
if not pts:
    raise SystemExit('no result-test probe lines in ' + sys.argv[2])
for (x0, v0), (x1, v1) in zip(pts, pts[1:]):
    if v0 >= 0.5 > v1:
        print('FRONT_%s=%.5f' % (sys.argv[3], x0 + (v0 - 0.5) / (v0 - v1) * (x1 - x0)))
        break
else:
    print('FRONT_%s=none' % sys.argv[3])
PYF

D=0.0001171
SI_DIFF=1.171e-10

python3 gen.py 40 $D        $D        $D        "1.0 0.0 0.0" consistent.yaml
python3 gen.py 40 $SI_DIFF  $SI_DIFF  $SI_DIFF  "1.0 0.0 0.0" siunits.yaml

probe CONSISTENT consistent.yaml
probe SIUNITS    siunits.yaml

echo "PROBE_LINES=$(grep -c 'is WRONG --> actresult=' "$TMP/CONSISTENT.log")"
echo "SIUNITS_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SIUNITS.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/SIUNITS.log"
python3 front.py consistent.yaml "$TMP/CONSISTENT.log" CONSISTENT
python3 front.py siunits.yaml    "$TMP/SIUNITS.log"    SIUNITS
python3 - "$TMP" <<'PY'
import subprocess, sys
def front(deck, log):
    out = subprocess.run(['python3', 'front.py', deck, log, 'X'],
                         capture_output=True, text=True).stdout.strip()
    return float(out.split('=')[1])
t = sys.argv[1]
c = front('consistent.yaml', t + '/CONSISTENT.log')
s = front('siunits.yaml',    t + '/SIUNITS.log')
x0 = 0.08 * 0.4
print("CONSISTENT_FRONT_TRAVEL=%.5f" % (c - x0))
print("SI_UNITS_FRONT_TRAVEL=%.5f" % (s - x0))
print("SI_UNITS_WAVE_PROPAGATED=%s" % ("yes" if s > x0 else "no"))
PY
echo "UNIT_WARNINGS=$(grep -ciE 'unit|dimension|scal(e|ing) error' "$TMP/SIUNITS.log")"
exit 0
