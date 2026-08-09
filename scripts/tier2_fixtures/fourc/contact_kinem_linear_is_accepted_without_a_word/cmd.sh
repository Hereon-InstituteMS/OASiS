#!/bin/bash
# Tier-2 for fourc::contact#5 — KINEM should be nonlinear for contact, and the
# reason it is a pitfall rather than a mistake is that 4C accepts the wrong
# choice without saying anything.
#
# Four arms: the same two-block mortar penalty deck at two load levels, each with
# KINEM nonlinear and KINEM linear. All four parse, build the contact interface,
# converge, and finish ten steps. Nothing in any log mentions kinematics.
#
# The measurement is the residual gap between the two facing faces — the
# penetration — read from 4C's report of the two facing corner nodes. Linear
# kinematics give a different answer, and the disagreement grows with the
# deformation: it roughly triples between the smaller and the larger load. That
# growth is the fingerprint, because at small deformation the two agree closely
# enough that a single run tells you nothing.
#
# Not asserted, because it is not what this deck shows: the entry's Hertz contact
# radius error and its "pressure distribution symmetric about the initial
# geometry". Those are post-processing statements about a problem this fixture
# does not solve. What it does show is that the error is silent and
# deformation-dependent.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = KINEM, $2 = prescribed z displacement
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, $2]
    FUNCT: [0, 0, 1]
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 5
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e-30
  - STRUCTURE:
      DIS: "structure"
      NODE: 9
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e-30
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
  - "NODE 8 DSURFACE 2"
  - "NODE 9 DSURFACE 3"
  - "NODE 10 DSURFACE 3"
  - "NODE 11 DSURFACE 3"
  - "NODE 12 DSURFACE 3"
  - "NODE 13 DSURFACE 4"
  - "NODE 14 DSURFACE 4"
  - "NODE 15 DSURFACE 4"
  - "NODE 16 DSURFACE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
  - "NODE 9 COORD 0.0 0.0 1.1"
  - "NODE 10 COORD 1.0 0.0 1.1"
  - "NODE 11 COORD 1.0 1.0 1.1"
  - "NODE 12 COORD 0.0 1.0 1.1"
  - "NODE 13 COORD 0.0 0.0 2.1"
  - "NODE 14 COORD 1.0 0.0 2.1"
  - "NODE 15 COORD 1.0 1.0 2.1"
  - "NODE 16 COORD 0.0 1.0 2.1"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM $1"
  - "2 SOLID HEX8 9 10 11 12 13 14 15 16 MAT 1 KINEM $1"
YAML
}

deck nonlinear "-0.3" > "$TMP/nl_small.yaml"
deck linear    "-0.3" > "$TMP/lin_small.yaml"
deck nonlinear "-0.6" > "$TMP/nl_large.yaml"
deck linear    "-0.6" > "$TMP/lin_large.yaml"

probe NLSMALL  "$TMP/nl_small.yaml"
probe LINSMALL "$TMP/lin_small.yaml"
probe NLLARGE  "$TMP/nl_large.yaml"
probe LINLARGE "$TMP/lin_large.yaml"

# All four run to the end and build a contact interface.
for a in NLSMALL LINSMALL NLLARGE LINLARGE; do
  echo "STEPS_$a=$(grep -c 'Finalised step' "$TMP/$a.log")"
done
grep -m1 -F "Building contact interface" "$TMP/LINLARGE.log"
grep -m1 -F "Contact-Normal-Active-Set-Size = 4" "$TMP/LINLARGE.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/LINLARGE.log"

python3 - "$TMP/NLSMALL.log" "$TMP/LINSMALL.log" "$TMP/NLLARGE.log" "$TMP/LINLARGE.log" <<'PY'
import re, sys
GAP = 0.1

def pen(path):
    t = open(path, "rb").read().decode("utf-8", "replace")
    v = {}
    for m in re.finditer(r'dispz +at node +(\d+)\s+is WRONG --> actresult=\s*([-0-9.e+]+),', t):
        v[m.group(1)] = float(m.group(2))
    assert {"5", "9"} <= set(v), "4C did not report both facing nodes for " + path
    return -(GAP + v["9"] - v["5"])

nls, lins, nll, linl = (pen(p) for p in sys.argv[1:5])
d_small = abs(lins - nls) / abs(nls)
d_large = abs(linl - nll) / abs(nll)
print("PENETRATION_NONLINEAR_SMALL=%.6e" % nls)
print("PENETRATION_LINEAR_SMALL=%.6e" % lins)
print("PENETRATION_NONLINEAR_LARGE=%.6e" % nll)
print("PENETRATION_LINEAR_LARGE=%.6e" % linl)
print("LINEAR_DIFFERS_AT_SMALL_LOAD=%s" % ("yes" if d_small > 1e-6 else "no"))
print("LINEAR_DIFFERS_AT_LARGE_LOAD=%s" % ("yes" if d_large > 1e-6 else "no"))
print("DISAGREEMENT_GROWS_WITH_DEFORMATION=%s" % ("yes" if d_large > 2.0 * d_small else "no"))
PY

# Not one word about kinematics in any of the four logs.
python3 - "$TMP/NLSMALL.log" "$TMP/LINSMALL.log" "$TMP/NLLARGE.log" "$TMP/LINLARGE.log" <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += (t.count("kinem") + t.count("linear kinematic")
          + t.count("geometrically linear") + t.count("small strain"))
print("KINEM_WARNINGS=%d" % n)
PY
exit 0
