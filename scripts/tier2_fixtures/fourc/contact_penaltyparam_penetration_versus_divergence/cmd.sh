#!/bin/bash
# Tier-2 for fourc::contact#2 — PENALTYPARAM really does trade penetration
# against solvability, and this fixture measures both ends on one deck.
#
# Two unit cubes, 0.1 apart, the upper one pushed down 0.3 over ten steps, mortar
# penalty contact. The gap that remains open is the penetration; it is computed
# from 4C's own report of the two facing corner nodes (5 on the lower body's top
# face, 9 on the upper body's bottom face), forced to print by an unmatchable
# RESULT DESCRIPTION.
#
#   PENALTYPARAM   penetration / element edge
#   1e0            ~20 %          contact barely resists at all
#   1e2            ~17 %
#   1e3            ~6 %           the entry's recommended starting point
#   1e4            ~0.8 %
#   1e6 and above  no answer:     "The nonlinear solver did not converge!"
#
# Two things to correct in the entry. Its own acceptance threshold is "max
# penetration exceeds ~5 % of the contact-pair element edge length" — and its own
# recommended starting value of 1e3 does not meet it on this problem, so "start
# with 1e3" is not a safe default, it is a first bracket. And the stiff end gives
# no warning of any kind: CONDITION_NUMBER_WARNINGS=0, so the promised
# "condition-number warning above ~1e14" is not an observable. What you get is a
# NOX abort.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = PENALTYPARAM
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
  PENALTYPARAM: $1
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
    VAL: [0.0, 0.0, -0.3]
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
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
  - "2 SOLID HEX8 9 10 11 12 13 14 15 16 MAT 1 KINEM nonlinear"
YAML
}

deck "1.0e0" > "$TMP/p0.yaml"
deck "1.0e2" > "$TMP/p2.yaml"
deck "1.0e3" > "$TMP/p3.yaml"
deck "1.0e4" > "$TMP/p4.yaml"
deck "1.0e6" > "$TMP/p6.yaml"

probe P0 "$TMP/p0.yaml"
probe P2 "$TMP/p2.yaml"
probe P3 "$TMP/p3.yaml"
probe P4 "$TMP/p4.yaml"
probe P6 "$TMP/p6.yaml"

# Every soft arm converges — that is what makes under-penalised contact dangerous.
for a in P0 P2 P3 P4; do
  echo "STEPS_$a=$(grep -c 'Finalised step' "$TMP/$a.log")"
done
grep -m1 -F "is WRONG --> actresult=" "$TMP/P4.log"
grep -m1 -F "Contact-Normal-Active-Set-Size = 4" "$TMP/P4.log"

# ...and the stiff arm simply stops.
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/P6.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/P6.log"

python3 - "$TMP/P0.log" "$TMP/P2.log" "$TMP/P3.log" "$TMP/P4.log" "$TMP/P6.log" <<'PY'
import re, sys
GAP = 0.1     # initial separation of the two faces
EDGE = 1.0    # element edge length of the contact pair

def pen(path):
    t = open(path, "rb").read().decode("utf-8", "replace")
    v = {}
    for m in re.finditer(r'dispz +at node +(\d+)\s+is WRONG --> actresult=\s*([-0-9.e+]+),', t):
        v[m.group(1)] = float(m.group(2))
    if "5" not in v or "9" not in v:
        return None
    return -(GAP + v["9"] - v["5"])

labels = ["1e0", "1e2", "1e3", "1e4", "1e6"]
vals = [pen(p) for p in sys.argv[1:]]
for lab, p in zip(labels, vals):
    print("PENETRATION_FRACTION_%s=%s" % (lab, "no_solution" if p is None else "%.4f" % (p / EDGE)))
soft, mid, tuned = vals[0], vals[2], vals[3]
print("SOFTEST_PENETRATES_OVER_10PCT=%s" % ("yes" if soft / EDGE > 0.10 else "no"))
print("PENETRATION_FALLS_WITH_PENALTY=%s" % ("yes" if soft > mid > tuned else "no"))
# The entry's own 5 % acceptance threshold, applied to its own recommended start.
print("RECOMMENDED_1E3_MEETS_5PCT_RULE=%s" % ("yes" if mid / EDGE <= 0.05 else "no"))
PY

python3 - "$TMP/P6.log" <<'PY'
import sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace").lower()
print("CONDITION_NUMBER_WARNINGS=%d" % (t.count("condition number") + t.count("ill-condition")))
PY
exit 0
