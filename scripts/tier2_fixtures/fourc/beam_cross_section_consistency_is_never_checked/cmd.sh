#!/bin/bash
# Tier-2 for fourc::beams#7 — CROSSAREA and the second moments MOMIN2/MOMIN3/
# MOMINPOL of MAT_BeamReissnerElastHyper are four independent numbers. 4C never
# cross-checks them, so a section that is not a section of anything runs happily.
#
# Two arms on the same cantilever, loaded at the tip with both an axial (x) and a
# transverse (z) force so that EA and EI are both exercised:
#
#   CONSISTENT   r = 0.1 solid circle: A = pi r^2, I = pi r^4/4, J = pi r^4/2
#   INCONSISTENT the same I and J, but A = 1.0 — the "A = 1.0 but Iyy = 0.01 for
#                a 'circular' section" mistake the entry describes
#
# Nothing is reported. Both decks parse, converge and finish; grep the logs for
# any mention of the cross-section and you get zero hits in both. What changes is
# the answer: the axial response follows A, the bending response does not, so the
# tip x-displacement flips sign while the z-displacement barely moves. The
# deliberately unmatchable RESULT DESCRIPTION (VALUE 0, TOLERANCE 1e-30) is what
# makes 4C print both numbers, in its own words, for both arms.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = CROSSAREA
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1
  TOLRES: 1e-06
  MAXITER: 25
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [1, 0, 1, 0, 0, 0]
    VAL: [200, 0, 2, 0, 0, 0]
    FUNCT: [1, 0, 1, 0, 0, 0]
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
  - "NODE 6 DNODE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 2.0 0.0 0.0"
  - "NODE 3 COORD 4.0 0.0 0.0"
  - "NODE 4 COORD 6.0 0.0 0.0"
  - "NODE 5 COORD 8.0 0.0 0.0"
  - "NODE 6 COORD 10.0 0.0 0.0"
STRUCTURE ELEMENTS:
  - "1 BEAM3R LINE2 1 2 MAT 1 TRIADS 0 0 0 0 0 0"
  - "2 BEAM3R LINE2 2 3 MAT 1 TRIADS 0 0 0 0 0 0"
  - "3 BEAM3R LINE2 3 4 MAT 1 TRIADS 0 0 0 0 0 0"
  - "4 BEAM3R LINE2 4 5 MAT 1 TRIADS 0 0 0 0 0 0"
  - "5 BEAM3R LINE2 5 6 MAT 1 TRIADS 0 0 0 0 0 0"
MATERIALS:
  - MAT: 1
    MAT_BeamReissnerElastHyper:
      YOUNG: 1e+07
      SHEARMOD: 5e+06
      DENS: 1.0
      CROSSAREA: $1
      SHEARCORR: 1
      MOMINPOL: 1.5707963267948968e-04
      MOMIN2: 7.853981633974484e-05
      MOMIN3: 7.853981633974484e-05
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 6
      QUANTITY: "dispx"
      VALUE: 0.0
      TOLERANCE: 1.0e-30
  - STRUCTURE:
      DIS: "structure"
      NODE: 6
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e-30
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_TIME: "t"
YAML
}

deck "0.031415926535897934" > "$TMP/consistent.yaml"
deck "1.0"                  > "$TMP/inconsistent.yaml"

probe CONSISTENT   "$TMP/consistent.yaml"
probe INCONSISTENT "$TMP/inconsistent.yaml"

# Both arms complete all ten steps.
echo "CONSISTENT_STEPS=$(grep -c 'Finalised step' "$TMP/CONSISTENT.log")"
echo "INCONSISTENT_STEPS=$(grep -c 'Finalised step' "$TMP/INCONSISTENT.log")"
# 4C never comments on the section at all.
echo "CROSS_SECTION_WARNINGS=$(cat "$TMP/CONSISTENT.log" "$TMP/INCONSISTENT.log" | grep -ciE 'cross.?section|inconsistent section|MOMIN')"
# The tip displacements, printed by 4C itself.
grep -m1 -F "dispx" "$TMP/CONSISTENT.log"
grep -m1 -F "dispx" "$TMP/INCONSISTENT.log"

python3 - "$TMP/CONSISTENT.log" "$TMP/INCONSISTENT.log" <<'PY'
import re, sys
def tip(path):
    t = open(path).read()
    v = {}
    for m in re.finditer(r'disp([xz]) +at node +\d+\s+is WRONG --> actresult=\s*([-0-9.e+]+),', t):
        v[m.group(1)] = float(m.group(2))
    return v
a, b = tip(sys.argv[1]), tip(sys.argv[2])
assert set(a) == set(b) == {"x", "z"}, "4C did not report both tip components"
print("CONSISTENT_TIP_DISPX=%.6e" % a["x"])
print("INCONSISTENT_TIP_DISPX=%.6e" % b["x"])
print("AXIAL_RESPONSE_SIGN_FLIPPED=%s" % ("yes" if a["x"] * b["x"] < 0 else "no"))
rel_z = abs(b["z"] - a["z"]) / abs(a["z"])
print("BENDING_RESPONSE_RELCHANGE_UNDER_1PCT=%s" % ("yes" if rel_z < 0.01 else "no"))
PY
exit 0
