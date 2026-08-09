#!/bin/bash
# Tier-2 for fourc::input_format#20 — a RESULT DESCRIPTION entry that names a
# thing which does not exist is never silently skipped, and the three ways of
# getting it wrong come from three different places in 4C:
#
#   wrong DIS      -> the entry matches nothing, and the result-test MANAGER
#                     catches the shortfall at the very end:
#                     "expected 1 tests but performed 0"   (utils_result_test)
#   wrong NODE     -> "Node 99999 does not belong to discretization structure"
#   wrong QUANTITY -> "Quantity 'banana' not supported in structure testing"
#                     (both from structure_new/.../4C_structure_new_resulttest.cpp)
#
# The first is the dangerous one: the simulation runs to completion, the
# mis-spelled field name is never echoed, and only a COUNT mismatch betrays it.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = DIS, $2 = NODE, $3 = QUANTITY, $4 = out file
cat > "$4" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 1
  MAXTIME: 0.1
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 1, 0]
    FUNCT: [0, 0, 0]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 1"
  - "NODE 8 DSURFACE 1"
  - "NODE 2 DSURFACE 2"
  - "NODE 3 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "$1"
      NODE: $2
      QUANTITY: "$3"
      VALUE: 4.47909266337460053e-03
      TOLERANCE: 1e-12
YAML
}

mk structure 3     dispy  "$TMP/good.yaml"
mk structur  3     dispy  "$TMP/dis.yaml"
mk structure 99999 dispy  "$TMP/node.yaml"
mk structure 3     banana "$TMP/qty.yaml"

probe GOOD "$TMP/good.yaml"
probe DIS  "$TMP/dis.yaml"
probe NODE "$TMP/node.yaml"
probe QTY  "$TMP/qty.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
grep -m1 -F "expected 1 tests but performed 0" "$TMP/DIS.log"
grep -m1 -F "4C_utils_result_test.cpp" "$TMP/DIS.log"
grep -m1 -F "Node 99999 does not belong to discretization structure" "$TMP/NODE.log"
grep -m1 -F "Quantity 'banana' not supported in structure testing" "$TMP/QTY.log"
grep -m1 -F "4C_structure_new_resulttest.cpp" "$TMP/QTY.log"
# The wrong-DIS run looks healthy right up to the count check: it never echoes
# the mis-spelled name and never reports a failing test.
echo "DIS_ARM_ECHOES_THE_TYPO=$(grep -c 'structur"' "$TMP/DIS.log")"
echo "DIS_ARM_REPORTS_A_WRONG_TEST=$(grep -c 'is WRONG' "$TMP/DIS.log")"
exit 0
