#!/bin/bash
# Tier-2 for fourc::tsi#3 — INITTEMP is the reference temperature for zero
# thermal strain, and it is a REQUIRED key of MAT_Struct_ThermoStVenantK.  One
# HEX8 SOLIDSCATRA cube in free expansion, THEXPANS 1.2e-05, L 1, thermal field
# clamped at 393 volume-wide.  Three arms:
#
#   REFERENCED  INITTEMP: 293  -> dispx = alpha*(393-293)*L = 1.2e-03   CORRECT
#   ZERO_REF    INITTEMP: 0    -> dispx = alpha*(393-0)*L   = 4.716e-03
#                                 exactly the "as if it started from absolute
#                                 zero" behaviour the claim describes
#   OMITTED     no INITTEMP    -> the deck does not parse
#
# The rule holds and the ZERO_REF arm measures it.  The Signal is false:
# "omitting INITTEMP defaults to 0" does not happen.  MAT_Struct_ThermoStVenantK
# lists INITTEMP as required, so the deck is rejected in MATERIALS before any
# element exists —
#     Failed to match specification in section 'MATERIALS'.  (4C_global_data_read.cpp)
#     Could not match this input                              (4C_io_input_spec_builders.cpp)
#     [X] Expected parameter 'INITTEMP'
# — which is a loud failure, not a silent default.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = the INITTEMP line (may be empty), $2 = prescribed T, $3 = expected dispx
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
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 1
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
$1      THERMOMAT: 2
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
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "$2"
DESIGN VOL THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [$2]
    FUNCT: [0]
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
DVOL-NODE TOPOLOGY:
  - "NODE 1 DVOL 1"
  - "NODE 2 DVOL 1"
  - "NODE 3 DVOL 1"
  - "NODE 4 DVOL 1"
  - "NODE 5 DVOL 1"
  - "NODE 6 DVOL 1"
  - "NODE 7 DVOL 1"
  - "NODE 8 DVOL 1"
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
      VALUE: $3
      TOLERANCE: 1e-12
YAML
}

deck "      INITTEMP: 293
" 393.0 0.0012   > "$TMP/referenced.yaml"
deck "      INITTEMP: 0
" 393.0 0.004716 > "$TMP/zero_ref.yaml"
deck ""           393.0 0.0012   > "$TMP/omitted.yaml"

probe REFERENCED "$TMP/referenced.yaml"
probe ZERO_REF   "$TMP/zero_ref.yaml"
probe OMITTED    "$TMP/omitted.yaml"

# INITTEMP 293 gives the expansion from the stress-free state ...
grep -m1 -F "dispx    at node   2	 is CORRECT" "$TMP/REFERENCED.log"
# ... and INITTEMP 0 gives the absolute-zero-referenced one, 3.93x larger.
grep -m1 -F "dispx    at node   2	 is CORRECT" "$TMP/ZERO_REF.log"
echo "ZERO_REF_RATIO_IS_393_OVER_100=$(python3 -c 'print(abs(0.004716/0.0012 - 3.93) < 1e-12)')"
# Omitting the key is a parse error, not a default.
grep -m1 -F "Failed to match specification in section 'MATERIALS'" "$TMP/OMITTED.log"
grep -m1 -F "Could not match this input" "$TMP/OMITTED.log"
grep -m1 -F "Expected parameter 'INITTEMP'" "$TMP/OMITTED.log"
grep -m1 -oF "4C_io_input_spec_builders.cpp" "$TMP/OMITTED.log"
echo "OMITTED_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/OMITTED.log")"
exit 0
