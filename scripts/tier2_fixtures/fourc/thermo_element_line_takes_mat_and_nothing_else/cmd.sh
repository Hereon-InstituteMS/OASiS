#!/bin/bash
# Tier-2 for fourc::thermal#2 — a THERMO element line is
#
#     <id> THERMO <celltype> <node ids...> MAT <id>
#
# and MAT is the ONLY keyword it accepts.  KINEM, TYPE and THICK — all of them
# legal on other 4C element categories — are fatal here, not ignored.  The
# diagnostic quotes the leftover text and then lists what it did understand,
# which is the useful half:
#
#     After parsing, the line still contains 'KINEM linear'.
#     Parsed parameters: MAT : 1
#
# Same deck four times; only the tail of the element line differs.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra tokens appended to the element line
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [100.0]
    FUNCT: [0]
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
THERMO ELEMENTS:
  - "1 THERMO HEX8 1 2 3 4 5 6 7 8 MAT 1 $1"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [1.0]
RESULT DESCRIPTION:
  - THERMAL:
      DIS: "thermo"
      NODE: 2
      QUANTITY: "temp"
      VALUE: 100.0
      TOLERANCE: 1e-8
YAML
}

deck ""                > "$TMP/mat_only.yaml"
deck "KINEM linear"    > "$TMP/kinem.yaml"
deck "TYPE Undefined"  > "$TMP/type.yaml"
deck "THICK 1.0"       > "$TMP/thick.yaml"

probe MAT_ONLY "$TMP/mat_only.yaml"
probe KINEM    "$TMP/kinem.yaml"
probe TYPE     "$TMP/type.yaml"
probe THICK    "$TMP/thick.yaml"

grep -m1 -F "is CORRECT" "$TMP/MAT_ONLY.log"
grep -m1 -F "After parsing, the line still contains 'KINEM linear'." "$TMP/KINEM.log"
grep -m1 -F "After parsing, the line still contains 'TYPE Undefined'." "$TMP/TYPE.log"
grep -m1 -F "After parsing, the line still contains 'THICK 1.0'." "$TMP/THICK.log"
grep -m1 -oF "Parsed parameters: MAT : 1" "$TMP/KINEM.log"
grep -m1 -oF "4C_io_input_spec.cpp" "$TMP/KINEM.log"
# The extra token is rejected, never silently absorbed: no run reached a result.
echo "KINEM_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/KINEM.log")"
exit 0
