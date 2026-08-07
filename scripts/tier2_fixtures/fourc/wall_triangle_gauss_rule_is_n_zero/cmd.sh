#!/bin/bash
# Tier-2 for fourc::structural_mechanics#1 — on the legacy WALL element the GP
# pair means different things on triangles and on quadrilaterals, and each
# family rejects the other family's spelling with its OWN message.
#
#   WALL TRI3/TRI6 + "GP 3 0"  -> runs
#   WALL TRI3/TRI6 + "GP 3 3"  -> Unknown number of Gauss points for tri element
#   WALL QUAD4     + "GP 3 3"  -> runs
#   WALL QUAD4     + "GP 3 0"  -> Insufficient number of Gauss points
#
# So "GP <n> <n>" is not a safe default you can write everywhere: it is fatal on
# a triangle and required on a quad.  Both throws come from w1/4C_w1_input.cpp.
#
# EAS is the second triangle restriction and it is 4-node-only.  'EAS full'
# names the cell type it was asked for, so TRI3 and TRI6 give different text:
# 'eas-technology not implemented for tri3 elements' and '... tri6 elements'.
#
# One self-contained 2D deck (inline nodes, no mesh file), unit square, left
# edge clamped, right edge line-Neumann.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = STRUCTURE ELEMENTS block
cat <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
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
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 2
    ONOFF: [0, 1]
    VAL: [0.0, 1.0]
    FUNCT: [0, 1]
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
  - "NODE 8 DLINE 1"
  - "NODE 2 DLINE 2"
  - "NODE 3 DLINE 2"
  - "NODE 6 DLINE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.5 0.0 0.0"
  - "NODE 6 COORD 1.0 0.5 0.0"
  - "NODE 7 COORD 0.5 1.0 0.0"
  - "NODE 8 COORD 0.0 0.5 0.0"
  - "NODE 9 COORD 0.5 0.5 0.0"
STRUCTURE ELEMENTS:
$1
YAML
}

arm() {  # $1 = label, $2 = STRUCTURE ELEMENTS block
  deck "$2" > "$TMP/$1.4C.yaml"
  probe "$1" "$TMP/$1.4C.yaml"
}

TRI3='  - "1 WALL TRI3 1 2 3 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"
  - "2 WALL TRI3 1 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"'
TRI3_BAD='  - "1 WALL TRI3 1 2 3 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"
  - "2 WALL TRI3 1 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"'
TRI3_EAS='  - "1 WALL TRI3 1 2 3 MAT 1 KINEM nonlinear EAS full THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"
  - "2 WALL TRI3 1 3 4 MAT 1 KINEM nonlinear EAS full THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"'
TRI6='  - "1 WALL TRI6 1 2 3 5 6 9 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"
  - "2 WALL TRI6 1 3 4 9 7 8 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"'
TRI6_BAD='  - "1 WALL TRI6 1 2 3 5 6 9 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"
  - "2 WALL TRI6 1 3 4 9 7 8 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"'
TRI6_EAS='  - "1 WALL TRI6 1 2 3 5 6 9 MAT 1 KINEM nonlinear EAS full THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"
  - "2 WALL TRI6 1 3 4 9 7 8 MAT 1 KINEM nonlinear EAS full THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"'
QUAD_NN='  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 3"'
QUAD_N0='  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 3 0"'

arm TRI3_GP30  "$TRI3"
arm TRI3_GP33  "$TRI3_BAD"
arm TRI3_EAS   "$TRI3_EAS"
arm TRI6_GP30  "$TRI6"
arm TRI6_GP33  "$TRI6_BAD"
arm TRI6_EAS   "$TRI6_EAS"
arm QUAD4_GP33 "$QUAD_NN"
arm QUAD4_GP30 "$QUAD_N0"

# The two arms that run really do assemble and solve.
grep -m1 -F "processor 0 finished normally" "$TMP/TRI3_GP30.log"
grep -m1 -F "processor 0 finished normally" "$TMP/QUAD4_GP33.log"

# The triangle Gauss rule: both TRI orders reject the square spelling, and the
# throw is the element input reader, not the solver.
grep -m1 -F "Unknown number of Gauss points for tri element" "$TMP/TRI3_GP33.log"
grep -m1 -F "Unknown number of Gauss points for tri element" "$TMP/TRI6_GP33.log"
grep -m1 -F "4C_w1_input.cpp" "$TMP/TRI3_GP33.log"

# The quad rejects the triangle spelling with a DIFFERENT message, which is why
# neither form is a safe default.
grep -m1 -F "Insufficient number of Gauss points" "$TMP/QUAD4_GP30.log"

# EAS is 4-node-only and the message names the cell type it was handed.
grep -m1 -F "eas-technology not implemented for tri3 elements" "$TMP/TRI3_EAS.log"
grep -m1 -F "eas-technology not implemented for tri6 elements" "$TMP/TRI6_EAS.log"

# Neither triangle failure mentions the GP keyword or EAS in the other's terms.
python3 - "$TMP/TRI3_GP33.log" "$TMP/QUAD4_GP30.log" <<'PY'
import sys
tri, quad = (open(p, "rb").read().decode("utf-8", "replace") for p in sys.argv[1:3])
print("TRI_AND_QUAD_SHARE_A_MESSAGE=%s"
      % ("yes" if "Unknown number of Gauss points" in quad else "no"))
print("QUAD_MESSAGE_IN_TRI_LOG=%d" % tri.count("Insufficient number of Gauss points"))
PY
exit 0
