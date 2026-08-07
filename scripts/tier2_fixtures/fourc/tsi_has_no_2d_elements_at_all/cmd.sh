#!/bin/bash
# Tier-2 for fourc::tsi#0 — 4C has no 2D TSI elements.  A minimal
# PROBLEMTYPE: Thermo_Structure_Interaction deck with PROBLEM SIZE/DIM 2 and one
# QUAD4 dies for every element category you might reach for, while the same
# deck in 3D with a HEX8 SOLIDSCATRA runs.
#
#   WALL QUAD4        -> 'Unsupported solid element type!'  4C_tsi_utils.cpp
#   SOLID QUAD4       -> "Element 'SOLID' does not seem to know cell type 'quad4'."
#   SOLIDSCATRA QUAD4 -> "Element 'SOLIDSCATRA' does not seem to know cell type 'quad4'."
#   SOLIDSCATRA HEX8 (3D control) -> runs, result test CORRECT, exit 0
#
# The rule holds.  One of the two quoted signals does not.  The claim said the
# WALL arm aborts with 'Invalid type of material law for wall element' from
# 4C_w1_mat.cpp:179.  That string exists in the source but is never reached:
# TSI's own clone strategy rejects a non-SolidScatra element long before any
# material is evaluated, so what you actually get is
# 'Unsupported solid element type!' from src/tsi/4C_tsi_utils.cpp — a message
# that names neither the material nor the WALL element.  Both the real text and
# the absence of the claimed one are asserted.
. "$(dirname "$0")/../_lib/preamble.sh"

deck2d() {  # $1 = the element line body
cat <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  MAXTIME: 0.2
  LINEAR_SOLVER: 2
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 0.1
  MAXTIME: 0.2
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 0.2
  TIMESTEP: 0.1
  ITEMAX: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [1e+11]
      NUE: 0
      DENS: 1
      THEXPANS: 1e-05
      INITTEMP: 293
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: 420
      CONDUCT:
        constant: [52]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "393.0"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "1 $1"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 2
      QUANTITY: "dispx"
      VALUE: 0.001
      TOLERANCE: 1e-09
YAML
}

deck2d "WALL QUAD4 1 2 3 4 MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2" > "$TMP/wall.yaml"
deck2d "SOLID QUAD4 1 2 3 4 MAT 1 KINEM linear"                                                    > "$TMP/solid.yaml"
deck2d "SOLIDSCATRA QUAD4 1 2 3 4 MAT 1 KINEM linear TYPE Undefined"                               > "$TMP/solidscatra.yaml"

probe WALL_2D        "$TMP/wall.yaml"
probe SOLID_2D       "$TMP/solid.yaml"
probe SOLIDSCATRA_2D "$TMP/solidscatra.yaml"

# The 3D control: the same physics in 3D with a HEX8 SOLIDSCATRA runs.
CTRL=$(upstream tsi_lincompression_1waydisp.4C.yaml) || exit 3
grep -q "SOLIDSCATRA HEX8" "$CTRL" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$CTRL" "$TMP/ctrl.yaml"
probe SOLIDSCATRA_3D "$TMP/ctrl.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SOLIDSCATRA_3D.log"
grep -m1 -F "Unsupported solid element type!" "$TMP/WALL_2D.log"
grep -m1 -oF "4C_tsi_utils.cpp" "$TMP/WALL_2D.log"
grep -m1 -F "Element 'SOLID' does not seem to know cell type 'quad4'." "$TMP/SOLID_2D.log"
grep -m1 -F "Element 'SOLIDSCATRA' does not seem to know cell type 'quad4'." "$TMP/SOLIDSCATRA_2D.log"
grep -m1 -oF "4C_fem_general_element_definition.cpp" "$TMP/SOLIDSCATRA_2D.log"
# The catalogued WALL signal is not what happens.
echo "CLAIMED_W1_MAT_MESSAGE=$(grep -ci 'Invalid type of material law for wall element' "$TMP/WALL_2D.log")"
echo "CLAIMED_W1_MAT_FILE=$(grep -ci '4C_w1_mat.cpp' "$TMP/WALL_2D.log")"
# The real one names neither the material nor the element category.
echo "WALL_DIAGNOSTIC_NAMES_WALL=$(grep -c 'Unsupported solid element type!.*WALL' "$TMP/WALL_2D.log")"
exit 0
