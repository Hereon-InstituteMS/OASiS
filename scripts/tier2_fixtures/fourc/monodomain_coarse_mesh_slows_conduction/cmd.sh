#!/bin/bash
# Tier-2 for fourc::cardiac_monodomain#1 — mesh size changes the conduction
# velocity by tens of percent and 4C never mentions it.
#
# Claim: "Mesh resolution must resolve the AP wavefront ... h > 1 mm gives visibly
#        stepped propagation ...; conduction velocity is also wrong by 10-30%."
# The upstream Cardiac_Monodomain decks are single elements, so this needs a real
# propagating wave. The fixture builds one: a HEX8 strip of length 0.4 with the
# same MAT_myocard FHN material and DIFF as scatra_myocard_FHN_material, excited
# over the first 8% of its length, integrated to t = 80 with dt = 0.1. The front
# position is measured from 4C's own result-test lines -- one probe per node, each
# with an impossible target so 4C prints "actresult=" -- by linear interpolation
# of the phi = 0.5 crossing.
# Observed: h = 0.005 puts the front at 0.19586 and h = 0.04, eight times coarser,
# at 0.14006. Net of the 0.032-long excited region that is a conduction velocity
# 34% too low, well inside the claimed band, and there is no mesh-resolution
# warning anywhere in the log.
. "$(dirname "$0")/../_lib/preamble.sh"

FHN=$(upstream scatra_myocard_FHN_material.4C.yaml) || exit 3
grep -q '      DIFF1: 0.0001171' "$FHN" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '      MODEL: "FHN"' "$FHN"     || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cd "$TMP" || exit 3

COARSE_NX=10

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
python3 gen.py 80            $D $D $D "1.0 0.0 0.0" fine.yaml
python3 gen.py "$COARSE_NX"  $D $D $D "1.0 0.0 0.0" coarse.yaml

probe FINE   fine.yaml
probe COARSE coarse.yaml

echo "FINE_PROBE_LINES=$(grep -c 'is WRONG --> actresult=' "$TMP/FINE.log")"
echo "COARSE_PROBE_LINES=$(grep -c 'is WRONG --> actresult=' "$TMP/COARSE.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/FINE.log"
python3 front.py fine.yaml   "$TMP/FINE.log"   FINE
python3 front.py coarse.yaml "$TMP/COARSE.log" COARSE
python3 - "$TMP" <<'PY'
import subprocess, sys
def front(deck, log):
    out = subprocess.run(['python3', 'front.py', deck, log, 'X'],
                         capture_output=True, text=True).stdout.strip()
    return float(out.split('=')[1])
t = sys.argv[1]
f = front('fine.yaml', t + '/FINE.log')
c = front('coarse.yaml', t + '/COARSE.log')
x0 = 0.08 * 0.4
print("CONDUCTION_VELOCITY_ERROR_PERCENT=%.0f" % (100.0 * ((c - x0) - (f - x0)) / (f - x0)))
PY
echo "MESH_RESOLUTION_WARNINGS=$(grep -ciE 'mesh resolution|element size|too coarse|under-?resolv' "$TMP/COARSE.log")"
exit 0
