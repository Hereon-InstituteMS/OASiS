#!/bin/bash
# Tier-2 for fourc::solid_mechanics#4 — which element type owns 2D structural
# cells, probed rather than assumed.  The two spellings share no keywords, so
# you cannot hedge by writing both.
#
#   WALL  QUAD4 <n..> MAT m KINEM k EAS e THICK t STRESS_STRAIN s GP a b
#   SOLID QUAD4 <n..> MAT m KINEM k THICKNESS t PLANE_ASSUMPTION p
#
# On this build WALL owns 2D.  The fixture asserts the whole failure surface:
# the full WALL line runs, SOLID QUAD4 is rejected by cell type, and a WALL line
# that borrows SOLID's shorter key set is rejected for a MISSING REQUIRED key —
# a different message from a different source file, which is what tells you the
# element type was right and the key list was wrong.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = element line body, $2 = output file
cat > "$2" <<YAML
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
    VAL: [0.0, -1.0]
    FUNCT: [0, 1]
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
  - "NODE 2 DLINE 2"
  - "NODE 3 DLINE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 4.0 0.0 0.0"
  - "NODE 3 COORD 4.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "1 $1"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: -3.69053979396462145e-02
      TOLERANCE: 1.0e-10
YAML
}

deck "WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_stress GP 2 2" "$TMP/wall.4C.yaml"
deck "SOLID QUAD4 1 2 3 4 MAT 1 KINEM nonlinear THICKNESS 1.0 PLANE_ASSUMPTION plane_stress"        "$TMP/solid.4C.yaml"
deck "WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear"                                                     "$TMP/short.4C.yaml"

probe WALL       "$TMP/wall.4C.yaml"
probe SOLID2D    "$TMP/solid.4C.yaml"
probe WALL_SHORT "$TMP/short.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WALL.log"
grep -m1 -F "Element 'SOLID' does not seem to know cell type 'quad4'." "$TMP/SOLID2D.log"
grep -m1 -F "4C_fem_general_element_definition.cpp" "$TMP/SOLID2D.log"
grep -m1 -F "Required value 'STRESS_STRAIN' not found in input line" "$TMP/WALL_SHORT.log"
# The SOLID diagnostic names the CELL type, never the word "2D" or "WALL":
# grepping it for the thing you should have written finds nothing.
echo "SOLID_DIAGNOSTIC_NAMES_WALL=$(grep -ci 'wall' "$TMP/SOLID2D.log")"
exit 0
