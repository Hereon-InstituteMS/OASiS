#!/bin/bash
# Tier-2 for fourc::contact#12 — a 2D contact deck that omits
# 'PROBLEM SIZE:\n  DIM: 2' fails deep inside the THREE-dimensional mortar
# coupling, so the message points at geometry instead of at the missing key.
#
# One two-block 2D deck (two unit squares 0.1 apart, WALL QUAD4, DESIGN LINE
# MORTAR CONTACT CONDITIONS 2D), run twice, the PROBLEM SIZE section the only
# difference:
#
#   with DIM: 2 -> ten steps, exit 0
#   without     -> 'auxiliary_plane called for unknown element type' from
#                  contact/4C_contact_coupling3d.cpp, exit 1
#
# The deck parses, the elements are built and the search starts; only when the
# 3D coupling tries to build an auxiliary plane for a line segment does it stop.
# Nothing in the message names DIM or PROBLEM SIZE.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = PROBLEM SIZE block (possibly empty), $2 = out
cat > "$2" <<YAML
$1PROBLEM TYPE:
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
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 4
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, -0.3]
    FUNCT: [0, 1]
DESIGN LINE MORTAR CONTACT CONDITIONS 2D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 2 DLINE 1"
  - "NODE 3 DLINE 2"
  - "NODE 4 DLINE 2"
  - "NODE 5 DLINE 3"
  - "NODE 6 DLINE 3"
  - "NODE 7 DLINE 4"
  - "NODE 8 DLINE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 1.1 0.0"
  - "NODE 6 COORD 1.0 1.1 0.0"
  - "NODE 7 COORD 1.0 2.1 0.0"
  - "NODE 8 COORD 0.0 2.1 0.0"
STRUCTURE ELEMENTS:
  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
  - "2 WALL QUAD4 5 6 7 8 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
YAML
}

DIM2='PROBLEM SIZE:
  DIM: 2
'
deck "$DIM2" "$TMP/with_dim.yaml"
deck ""      "$TMP/no_dim.yaml"

probe WITH_DIM "$TMP/with_dim.yaml"
probe NO_DIM   "$TMP/no_dim.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITH_DIM.log"
echo "STEPS_WITH_DIM=$(grep -c 'Finalised step' "$TMP/WITH_DIM.log")"

grep -m1 -F "auxiliary_plane called for unknown element type" "$TMP/NO_DIM.log"
grep -m1 -F "4C_contact_coupling3d.cpp" "$TMP/NO_DIM.log"
echo "STEPS_NO_DIM=$(grep -c 'Finalised step' "$TMP/NO_DIM.log")"

# The deck gets a long way in before it dies, and the message never names the
# key that is missing.
echo "NO_DIM_BUILT_THE_CONTACT_INTERFACE=$(grep -c 'Building contact interface' "$TMP/NO_DIM.log")"
python3 - "$TMP/NO_DIM.log" <<'PY'
import sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
line = [l for l in t.split("\n") if "auxiliary_plane called" in l][0]
print("MESSAGE_NAMES_THE_MISSING_KEY=%s"
      % ("yes" if ("DIM" in line or "PROBLEM SIZE" in line) else "no"))
print("MESSAGE_NAMES_A_DIMENSION=%s"
      % ("yes" if ("2D" in line or "3D" in line) else "no"))
PY
exit 0
