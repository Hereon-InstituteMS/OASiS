#!/bin/bash
# Tier-2 for fourc::cardiac_monodomain#2 — with isotropic DIFF the FIBER1 vector
# on the element line has literally no effect.
#
# Claim: "Isotropic diffusion (DIFF1 = DIFF2 = DIFF3) gives SPHERICAL propagation
#        (no fiber alignment) ... anisotropic physiology (DIFF1 ~ 4*DIFF2 for
#        myocardium) but isotropic DIFF in the MAT_myocard MATERIALS block gives
#        spherical instead of elliptical wavefronts."
# The upstream Cardiac_Monodomain decks are single elements, so the fixture builds
# a propagating wave: a HEX8 strip of length 0.4 with the same MAT_myocard FHN
# material as scatra_myocard_FHN_material, run once with the fiber along the
# direction of propagation and once across it. Front positions come from 4C's own
# result-test "actresult=" lines.
# Observed:
#   DIFF1 = DIFF2 = DIFF3 : front 0.19744 along fiber and 0.19744 across it --
#     bit-for-bit identical, so FIBER1 is inert and the wavefront is spherical.
#   DIFF1 = 4*DIFF2       : front 0.19744 along fiber and 0.12024 across it, a
#     64% difference, which is the elliptical wavefront the claim describes.
# Nothing is printed about fibers in either case.
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
D4=2.9275e-05
ACROSS_DIFF="$D4"

python3 gen.py 40 $D  $D          $D          "1.0 0.0 0.0" iso_along.yaml
python3 gen.py 40 $D  $D          $D          "0.0 1.0 0.0" iso_across.yaml
python3 gen.py 40 $D  "$ACROSS_DIFF" "$ACROSS_DIFF" "1.0 0.0 0.0" ani_along.yaml
python3 gen.py 40 $D  "$ACROSS_DIFF" "$ACROSS_DIFF" "0.0 1.0 0.0" ani_across.yaml

probe ISO_ALONG  iso_along.yaml
probe ISO_ACROSS iso_across.yaml
probe ANI_ALONG  ani_along.yaml
probe ANI_ACROSS ani_across.yaml

echo "PROBE_LINES=$(grep -c 'is WRONG --> actresult=' "$TMP/ISO_ALONG.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/ANI_ACROSS.log"
python3 front.py iso_along.yaml  "$TMP/ISO_ALONG.log"  ISO_ALONG
python3 front.py iso_across.yaml "$TMP/ISO_ACROSS.log" ISO_ACROSS
python3 front.py ani_along.yaml  "$TMP/ANI_ALONG.log"  ANI_ALONG
python3 front.py ani_across.yaml "$TMP/ANI_ACROSS.log" ANI_ACROSS
python3 - "$TMP" <<'PY'
import subprocess, sys
def front(deck, log):
    out = subprocess.run(['python3', 'front.py', deck, log, 'X'],
                         capture_output=True, text=True).stdout.strip()
    return float(out.split('=')[1])
t = sys.argv[1]
ia = front('iso_along.yaml',  t + '/ISO_ALONG.log')
ic = front('iso_across.yaml', t + '/ISO_ACROSS.log')
aa = front('ani_along.yaml',  t + '/ANI_ALONG.log')
ac = front('ani_across.yaml', t + '/ANI_ACROSS.log')
print("VERDICT: ISOTROPIC_DIFF_SEES_THE_FIBER=%s" % ("no" if ia == ic else "yes"))
print("ANISOTROPIC_FIBER_FRONT_ADVANTAGE_PERCENT=%.0f" % (100.0 * (aa - ac) / ac))
PY
echo "FIBER_WARNINGS=$(grep -ciE 'fiber|anisotrop' "$TMP/ISO_ALONG.log")"
exit 0
