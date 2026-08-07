#!/bin/bash
# Tier-2 for fourc::input_format#10 — which element TYPE owns 2D structural
# cells is a property of the build, and the two candidate spellings share no
# keywords, so you have to know which one this binary registers before writing
# a single 2D element line.
#
# On this build SOLID owns nine 3D cell types and nothing else, and QUAD4
# belongs to WALL with the keyword set MAT / KINEM / EAS / THICK /
# STRESS_STRAIN / GP.  The fixture reads that ownership straight out of
# `4C --parameters` and then proves it by running all three cases:
#
#   WALL QUAD4      -> runs
#   SOLID QUAD4     -> "Element 'SOLID' does not seem to know cell type 'quad4'."
#                      (cell type echoed in LOWERCASE, from
#                       core/fem/.../4C_fem_general_element_definition.cpp)
#   BOGUSELE QUAD4  -> "Unknown type 'BOGUSELE' of finite element"
#                      (from core/comm/src/4C_comm_parobjectfactory.cpp)
#
# Note the QUOTES in the second message: the template is
# "Unknown type '{}' of finite element", so an unquoted grep finds nothing.
# The two messages mean different things -- "no such element type at all"
# versus "that type exists but does not own that cell".
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = element line, $2 = out file
cat > "$2" <<YAML
PROBLEM SIZE:
  DIM: 2
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
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 2
    ONOFF: [0, 1]
    VAL: [0, 1]
    FUNCT: [0, 0]
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
  - "NODE 2 DLINE 2"
  - "NODE 3 DLINE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "$1"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: 4.33333333333345890e-03
      TOLERANCE: 1e-12
YAML
}

mk "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2" "$TMP/wall.yaml"
mk "1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM linear" "$TMP/solid.yaml"
mk "1 BOGUSELE QUAD4 1 2 3 4 MAT 1 KINEM linear" "$TMP/bogus.yaml"

probe WALL  "$TMP/wall.yaml"
probe SOLID "$TMP/solid.yaml"
probe BOGUS "$TMP/bogus.yaml"

grep -m1 -F "is CORRECT, abs(diff)= 0.00000000000000000e+00" "$TMP/WALL.log"
grep -m1 -F "processor 0 finished normally" "$TMP/WALL.log"
grep -m1 -F "Element 'SOLID' does not seem to know cell type 'quad4'." "$TMP/SOLID.log"
grep -m1 -F "4C_fem_general_element_definition.cpp" "$TMP/SOLID.log"
grep -m1 -F "Unknown type 'BOGUSELE' of finite element" "$TMP/BOGUS.log"
grep -m1 -F "4C_comm_parobjectfactory.cpp" "$TMP/BOGUS.log"
# The registered-but-wrong-cell case is NOT the unknown-type case.
echo "SOLID_ARM_SAID_UNKNOWN_TYPE=$(grep -c 'of finite element' "$TMP/SOLID.log")"

# Read the ownership out of the binary rather than out of a catalogue.
"$BIN" --parameters 2>/dev/null > "$TMP/params.yaml"
cells() {  # cell types owned by legacy element $1
  awk -v want="  $1:" '
    /^legacy_element_specs:$/{inls=1;next}
    /^[A-Za-z_$]/{inls=0}
    inls && $0==want {on=1;next}
    inls && on && /^  [A-Za-z0-9_]+:$/{on=0}
    inls && on && /^    - cell_type: /{print $3}' "$TMP/params.yaml"
}
echo "SCHEMA_SOLID_CELLS=$(cells SOLID | tr '\n' ' ')"
echo "SCHEMA_SOLID_OWNS_QUAD4=$(cells SOLID | grep -cx QUAD4)"
echo "SCHEMA_WALL_OWNS_QUAD4=$(cells WALL | grep -cx QUAD4)"
echo "SCHEMA_SOLID_OWNS_HEX8=$(cells SOLID | grep -cx HEX8)"
echo "SCHEMA_HAS_BOGUSELE=$(awk '/^legacy_element_specs:$/{f=1;next} /^[A-Za-z_$]/{f=0} f && /^  BOGUSELE:$/' "$TMP/params.yaml" | wc -l)"
exit 0
