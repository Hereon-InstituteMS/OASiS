#!/bin/bash
# Tier-2 for fourc::input_format#29 — the NUE validator is in_range[-1,0.5),
# and the bracket at the LOW end is CLOSED.  NUE: -1.0 therefore PASSES
# validation, produces no material diagnostic of any kind, and then dies of
# SIGFPE inside the element evaluation because the Lame constants blow up.
#
# That makes -1.0 the one admissible Poisson value that gets you neither a
# number nor a parse error.  The high end behaves the way anyone would expect:
# 0.5 and 0.6 are both rejected at parse with the validator string quoted.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = NUE, $2 = out file
cat > "$2" <<YAML
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
      NUE: $1
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: 3.48962414247091460e-03
      TOLERANCE: 1e-12
YAML
}

mk 0.49 "$TMP/inside.yaml"
mk 0.5  "$TMP/half.yaml"
mk 0.6  "$TMP/over.yaml"
mk -1.0 "$TMP/minus1.yaml"

probe INSIDE "$TMP/inside.yaml"
probe HALF   "$TMP/half.yaml"
probe OVER   "$TMP/over.yaml"
probe MINUS1 "$TMP/minus1.yaml"

# The useful end of the range: 0.49 is an ordinary run with an ordinary answer.
grep -m1 -F "is CORRECT, abs(diff)= 0.00000000000000000e+00" "$TMP/INSIDE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/INSIDE.log"
# The high end is closed against you, with the validator quoted verbatim.
grep -m1 -F "Candidate parameter 'NUE' does not pass validation: in_range[-1,0.5)" "$TMP/HALF.log"
grep -m1 -F "Candidate parameter 'NUE' does not pass validation: in_range[-1,0.5)" "$TMP/OVER.log"
grep -m1 -F "Failed to match specification in section 'MATERIALS'" "$TMP/HALF.log"
# The low end is OPEN TO -1.0: no validator complaint, no material diagnostic,
# no 4C error block at all -- just signal 8, which the shell reports as 136.
echo "MINUS1_FAILED_VALIDATION=$(grep -c "does not pass validation" "$TMP/MINUS1.log")"
echo "MINUS1_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/MINUS1.log")"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/MINUS1.log"
exit 0
