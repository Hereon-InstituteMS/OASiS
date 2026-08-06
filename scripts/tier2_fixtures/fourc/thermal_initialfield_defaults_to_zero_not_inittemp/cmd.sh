#!/bin/bash
# Tier-2 for fourc::thermal#7 — omitting THERMAL DYNAMIC/INITIALFIELD leaves
# the temperature at exactly T = 0, and for a structural material carrying
# INITTEMP > 0 that is a spurious thermal strain from the very first step,
# because it is the difference T - INITTEMP that drives the expansion.
#
# One HEX8 SOLIDSCATRA cube, symmetry-constrained on three faces, no thermal
# boundary condition at all, so the thermal field simply keeps its initial
# value.  INITTEMP = 293, THEXPANS = 1.2e-05, L = 1.  Two decks differing only
# in the presence of two lines:
#
#   INITIALFIELD: "field_by_function" + INITFUNCNO: 1 (FUNCT1 = 293.0)
#        -> T = 293 = INITTEMP, dispx = 0, unstrained, exit 0
#   both lines removed
#        -> T = 0, dispx = alpha*(0-293)*L = -3.516e-03, a contraction that
#           should not be there — and the run still exits 0 on a deck without a
#           result test.  There is no parse or run-time warning either way.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = the INITIALFIELD lines (may be empty)
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCT NOX/Printing:
  Inner Iteration: false
  Outer Iteration StatusTest: false
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  MAXTIME: 1
  TOLRES: 0.001
  TOLDISP: 1e-10
  LINEAR_SOLVER: 2
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
$1  TIMESTEP: 1
  MAXTIME: 1
  LINEAR_SOLVER: 1
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: "Temperature"
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 1
  TIMESTEP: 1
  NUMSTEP: 1
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
      YOUNG: [2.1e+11]
      NUE: 0
      DENS: 7850
      THEXPANS: 1.2e-05
      INITTEMP: 293
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: 3.297e+06
      CONDUCT:
        constant: [52]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "293.0"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 0, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 3
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 1"
  - "NODE 8 DSURFACE 1"
  - "NODE 1 DSURFACE 2"
  - "NODE 2 DSURFACE 2"
  - "NODE 5 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 1 DSURFACE 3"
  - "NODE 2 DSURFACE 3"
  - "NODE 3 DSURFACE 3"
  - "NODE 4 DSURFACE 3"
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
  - "1 SOLIDSCATRA HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM linear TYPE Undefined"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 2
      QUANTITY: "dispx"
      VALUE: 0.0
      TOLERANCE: 1e-12
  - THERMAL:
      DIS: "thermo"
      NODE: 7
      QUANTITY: "temp"
      VALUE: 293.0
      TOLERANCE: 1e-09
YAML
}

deck '  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
' > "$TMP/with.yaml"
deck '' > "$TMP/without.yaml"

probe WITH_INITIALFIELD    "$TMP/with.yaml"
probe WITHOUT_INITIALFIELD "$TMP/without.yaml"

grep -m1 -F "dispx    at node   2	 is CORRECT" "$TMP/WITH_INITIALFIELD.log"
grep -m1 -F "temp     at node   7	 is CORRECT" "$TMP/WITH_INITIALFIELD.log"
# Without the key the temperature is exactly zero, not INITTEMP ...
grep -m1 -F "temp     at node   7	 is WRONG --> actresult= 0.00000000000000000e+00" "$TMP/WITHOUT_INITIALFIELD.log"
# ... and the cube contracts by alpha*(0-INITTEMP)*L at t = 0.
grep -m1 -F "dispx    at node   2	 is WRONG --> actresult=-3.51599999999999959e-03" "$TMP/WITHOUT_INITIALFIELD.log"
# 4C says nothing about the missing key in either run.
echo "INITIALFIELD_DIAGNOSTIC=$(grep -ciE 'initialfield|initfuncno' "$TMP/WITHOUT_INITIALFIELD.log")"
echo "WITHOUT_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/WITHOUT_INITIALFIELD.log")"
exit 0
