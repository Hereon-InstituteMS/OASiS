#!/bin/bash
# Tier-2 for fourc::contact#1 — a mortar contact interface needs one Slave AND
# one Master carrying the same InterfaceID. The rule is right; the two log lines
# the entry quoted do not exist.
#
# Claimed:  'no master partner found for interface X' /
#           'MortarInterface: InterfaceID X has 0 master elements'
# Observed: neither string is in 4C. Both failure modes abort from
#           contact/src/4C_contact_utils.cpp, and which message you get depends
#           on how the group is malformed, not on which side is missing:
#
#     both surfaces marked Slave -> Master side missing in contact condition group!
#     only one surface listed    -> Not enough contact conditions in discretization
#
# The second is the trap: a single-sided condition never even reaches the
# side-checking code, so the message says nothing about Slave or Master.
#
# One self-contained two-block penalty deck, three condition blocks.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = the DESIGN SURF MORTAR CONTACT CONDITIONS 3D block
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
    VAL: [0.0, 0.0, -0.3]
    FUNCT: [0, 0, 1]
$1DSURF-NODE TOPOLOGY:
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

PAIRED='DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
'
TWO_SLAVES='DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Slave"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
'
ONE_SIDED='DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 3
    InterfaceID: 1
    Side: "Slave"
'

deck "$PAIRED"     > "$TMP/paired.yaml"
deck "$TWO_SLAVES" > "$TMP/two_slaves.yaml"
deck "$ONE_SIDED"  > "$TMP/one_sided.yaml"

probe PAIRED    "$TMP/paired.yaml"
probe TWOSLAVES "$TMP/two_slaves.yaml"
probe ONESIDED  "$TMP/one_sided.yaml"

# The control really does build an interface and press the blocks together.
grep -m1 -F "Building contact interface" "$TMP/PAIRED.log"
grep -m1 -F "processor 0 finished normally" "$TMP/PAIRED.log"
echo "PAIRED_ACTIVE_SET_NONZERO=$(grep -c 'Contact-Normal-Active-Set-Size = 4' "$TMP/PAIRED.log")"

grep -m1 -F "Master side missing in contact condition group!" "$TMP/TWOSLAVES.log"
grep -m1 -F "4C_contact_utils.cpp" "$TMP/TWOSLAVES.log"
grep -m1 -F "Not enough contact conditions in discretization" "$TMP/ONESIDED.log"

python3 - "$TMP/TWOSLAVES.log" "$TMP/ONESIDED.log" <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += t.count("no master partner found") + t.count("master elements") + t.count("mortarinterface:")
print("CLAIMED_INTERFACE_TEXTS=%d" % n)
PY
# The single-sided message never names a side, which is why it reads as a
# missing-section problem rather than a missing-partner problem.
python3 - "$TMP/ONESIDED.log" <<'PY'
import sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
line = [l for l in t.split("\n") if "Not enough contact conditions" in l][0]
print("ONESIDED_MESSAGE_NAMES_A_SIDE=%s"
      % ("yes" if ("Slave" in line or "Master" in line) else "no"))
PY
exit 0
