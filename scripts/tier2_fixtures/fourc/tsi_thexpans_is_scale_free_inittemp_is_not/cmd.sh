#!/bin/bash
# Tier-2 for fourc::tsi#2 — the 273.15 offset is real, but THEXPANS is not
# where it comes from.  One HEX8 SOLIDSCATRA cube, free thermal expansion,
# THEXPANS held at 1.2e-05 in ALL THREE arms; only the temperature SCALE moves.
#
#   ALL_KELVIN   INITTEMP 293    T 393     -> dispx 1.2e-03  CORRECT
#   ALL_CELSIUS  INITTEMP 19.85  T 119.85  -> dispx 1.2e-03  CORRECT
#   MIXED        INITTEMP 19.85  T 393     -> dispx 4.4778e-03, off by exactly
#                                             3.2778e-03 = alpha*273.15*L
#
# ALL_KELVIN and ALL_CELSIUS describe the same physical state and give the SAME
# displacement with the SAME alpha.  That is the falsification: a linear
# expansion coefficient in 1/K and in 1/degC is the same number, so "THEXPANS
# UNITS must match the temperature units" has nothing to rescale.  What must
# match is the temperature SCALE of INITTEMP and of the thermal field; mixing
# them adds alpha * 273.15 * L, which is what the claim measured and
# misattributed.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = INITTEMP, $2 = prescribed T, $3 = expected dispx
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
      INITTEMP: $1
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

# Every arm expects the SAME displacement: alpha * 100 * L = 1.2e-03.
deck 293   393.0   0.0012 > "$TMP/kelvin.yaml"
deck 19.85 119.85  0.0012 > "$TMP/celsius.yaml"
deck 19.85 393.0   0.0012 > "$TMP/mixed.yaml"

probe ALL_KELVIN  "$TMP/kelvin.yaml"
probe ALL_CELSIUS "$TMP/celsius.yaml"
probe MIXED       "$TMP/mixed.yaml"

# Same alpha, same answer, two different temperature scales.
grep -m1 -F "dispx    at node   2	 is CORRECT" "$TMP/ALL_KELVIN.log"
grep -m1 -F "dispx    at node   2	 is CORRECT" "$TMP/ALL_CELSIUS.log"
echo "THEXPANS_RESCALED_BETWEEN_ARMS=$(grep -h 'THEXPANS' "$TMP/kelvin.yaml" "$TMP/celsius.yaml" "$TMP/mixed.yaml" | sort -u | wc -l)"
# Mixing the scales adds exactly alpha * 273.15 * L = 3.2778e-03.
grep -m1 -F "dispx    at node   2	 is WRONG --> actresult= 4.47779999999999880e-03, givenresult= 1.19999999999999989e-03, abs(diff)= 3.27779999999999912e-03" "$TMP/MIXED.log"
python3 -c "print('OFFSET_IS_ALPHA_TIMES_273_15=%s' % (abs(1.2e-05*273.15*1.0 - 3.27779999999999912e-03) < 1e-15))"
exit 0
