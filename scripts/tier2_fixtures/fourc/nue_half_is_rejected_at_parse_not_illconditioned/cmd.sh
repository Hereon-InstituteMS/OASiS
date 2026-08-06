#!/bin/bash
# Tier-2 for fourc::structural_mechanics#12 — NUE is validated against the
# half-open interval [-1, 0.5), so NUE: 0.5 is REFUSED at parse rather than
# merely ill-conditioned.
#
# Six arms of one Statics HEX8 deck, KINEM linear, the material the only change:
#
#   NUE  0.3        -> exit 0
#   NUE  0.4999     -> exit 0                (near-incompressible is reachable)
#   NUE  0.5        -> exit 1 at parse, in_range[-1,0.5)
#   NUE  0.6        -> exit 1 at parse, in_range[-1,0.5)
#   NUE -1.0        -> ACCEPTED by the parser (the bracket is CLOSED at -1) and
#                      then killed by SIGFPE, shell status 136
#   NUE -1.000001   -> exit 1 at parse, in_range[-1,0.5)
#
# The -1.0 arm is what makes "half-open" concrete: the two ends of the interval
# behave differently, and only the upper one is excluded.  It also shows that
# passing validation is not the same as being a usable material.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 NUE, $2 out
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: $1
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0.0, 1.0, 0.0]
    FUNCT: [0, 1, 0]
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
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM linear"
YAML
}

deck 0.3        "$TMP/a.yaml"
deck 0.4999     "$TMP/b.yaml"
deck 0.5        "$TMP/c.yaml"
deck 0.6        "$TMP/d.yaml"
deck -1.0       "$TMP/e.yaml"
deck -1.000001  "$TMP/f.yaml"

probe NUE_0_3            "$TMP/a.yaml"
probe NUE_0_4999         "$TMP/b.yaml"
probe NUE_0_5            "$TMP/c.yaml"
probe NUE_0_6            "$TMP/d.yaml"
probe NUE_MINUS_1        "$TMP/e.yaml"
probe NUE_BELOW_MINUS_1  "$TMP/f.yaml"

# Both admissible values run all the way through.
grep -m1 -F "processor 0 finished normally" "$TMP/NUE_0_3.log"
grep -m1 -F "processor 0 finished normally" "$TMP/NUE_0_4999.log"

# The incompressible limit is a parse rejection, not a conditioning problem.
grep -m1 -F "Candidate parameter 'NUE' does not pass validation: in_range[-1,0.5)" "$TMP/NUE_0_5.log"
grep -m1 -F "Failed to match specification in section 'MATERIALS'" "$TMP/NUE_0_5.log"
grep -m1 -F "4C_global_data_read.cpp" "$TMP/NUE_0_5.log"
grep -m1 -F "Candidate parameter 'NUE' does not pass validation: in_range[-1,0.5)" "$TMP/NUE_0_6.log"
grep -m1 -F "Candidate parameter 'NUE' does not pass validation: in_range[-1,0.5)" "$TMP/NUE_BELOW_MINUS_1.log"

# Nothing is assembled in a rejected run...
echo "NUE_0_5_REACHED_FILL_COMPLETE=$(grep -c 'fill_complete() on discretization structure' "$TMP/NUE_0_5.log")"
# ...while the CLOSED lower end is admitted and then kills the process, which is
# how you can tell the interval really is half-open rather than symmetric.
echo "NUE_MINUS_1_REACHED_FILL_COMPLETE=$(grep -c 'fill_complete() on discretization structure' "$TMP/NUE_MINUS_1.log")"
echo "NUE_MINUS_1_VALIDATION_COMPLAINTS=$(grep -c "does not pass validation" "$TMP/NUE_MINUS_1.log")"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/NUE_MINUS_1.log"
exit 0
