#!/bin/bash
# Tier-2 for fourc::tsi#9 — DESIGN VOL THERMO DIRICH CONDITIONS + DVOL-NODE
# TOPOLOGY is what clamps a whole region's temperature; a surface-only set
# leaves the interior free to develop its own profile.
#
# One HEX8 SOLIDSCATRA cube, thermal field transient (OneStepTheta) starting
# from 293, driven to 393.  Two arms, identical except for which condition
# section carries the 393:
#
#   VOL   DESIGN VOL  THERMO DIRICH on DVOL 1 (all 8 nodes)
#         -> node 7 = exactly 3.93000000000000000e+02 and the cube expands by
#            the uniform alpha*(393-293)*L = 1.2e-03.  Both result tests pass.
#   SURF  DESIGN SURF THERMO DIRICH on DSURFACE 4 (the x=0 face only)
#         -> node 7 is still ~293 after the same time, the field is
#            non-uniform, and the expansion is a quarter of the uniform one.
#
# Nothing warns.  The surface arm is only caught because the RESULT DESCRIPTION
# entries pin the clamped answer.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = the thermal condition block
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
  DYNAMICTYPE: "OneStepTheta"
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
$1
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
  - "NODE 1 DSURFACE 4"
  - "NODE 4 DSURFACE 4"
  - "NODE 5 DSURFACE 4"
  - "NODE 8 DSURFACE 4"
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
      VALUE: 0.0012
      TOLERANCE: 1e-12
  - THERMAL:
      DIS: "thermo"
      NODE: 7
      QUANTITY: "temp"
      VALUE: 393.0
      TOLERANCE: 1e-09
YAML
}

VOL='DESIGN VOL THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [393.0]
    FUNCT: [0]'
SURF='DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [393.0]
    FUNCT: [0]'

deck "$VOL"  > "$TMP/vol.yaml"
deck "$SURF" > "$TMP/surf.yaml"

probe VOL_CLAMP  "$TMP/vol.yaml"
probe SURF_CLAMP "$TMP/surf.yaml"

# Volume Dirichlet: every node sits at the prescribed value, uniform expansion.
grep -m1 -F "temp     at node   7	 is CORRECT" "$TMP/VOL_CLAMP.log"
grep -m1 -F "dispx    at node   2	 is CORRECT" "$TMP/VOL_CLAMP.log"
# Surface-only: the interior is free and stays near its initial 293.
grep -m1 -F "temp     at node   7	 is WRONG --> actresult= 2.93004731462221912e+02" "$TMP/SURF_CLAMP.log"
grep -m1 -F "dispx    at node   2	 is WRONG --> actresult= 3.33374339339256561e-04" "$TMP/SURF_CLAMP.log"
# Both decks prescribe the same 393 and neither run is warned about anything.
echo "SURF_PRESCRIBES_SAME_VALUE=$(grep -c 'VAL: \[393.0\]' "$TMP/surf.yaml")"
echo "SURF_DIAGNOSTIC=$(grep -ciE 'interior|not clamped|only.*surface' "$TMP/SURF_CLAMP.log")"
echo "SURF_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SURF_CLAMP.log")"
exit 0
