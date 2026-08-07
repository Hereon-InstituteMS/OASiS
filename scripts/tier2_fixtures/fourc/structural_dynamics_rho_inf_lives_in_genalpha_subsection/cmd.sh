#!/bin/bash
# Tier-2 for fourc::structural_dynamics#3 — RHO_INF, where it lives and what it
# does.  It is NOT a key of STRUCTURAL DYNAMIC; it belongs to the nested
# section STRUCTURAL DYNAMIC/GENALPHA, and putting it one level up is a parse
# error, not a silently ignored line.
#
#   DEFAULT       no GENALPHA section at all      -> exit 0, banner "rho = 1"
#   RHO_ONE       GENALPHA: RHO_INF 1.0           -> exit 0, SAME answer, so 1.0
#                                                   really is the default
#   RHO_HALF      GENALPHA: RHO_INF 0.5           -> different answer, and 4C
#                                                   echoes the derived
#                                                   coefficients it computed
#   WRONG_SECTION RHO_INF under STRUCTURAL DYNAMIC -> "Could not match this input"
#
# 4C prints rho/beta/gamma/alpha_f/alpha_m at start-up, so the effective
# parameter set is readable straight out of the log.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra STRUCTURAL DYNAMIC line, $2 = GENALPHA block, $3 = out
cat > "$3" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:$1
  DYNAMICTYPE: "GenAlpha"
  TIMESTEP: 0.05
  NUMSTEP: 8
  MAXTIME: 0.4
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-08
  MAXITER: 30
  LINEAR_SOLVER: 1
$2SOLVER 1:
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
    VAL: [0.0, 10.0, 0.0]
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
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: 1.70856951294205382e-02
      TOLERANCE: 1.0e-09
YAML
}

deck ""                  ""                                                "$TMP/default.4C.yaml"
deck ""                  $'STRUCTURAL DYNAMIC/GENALPHA:\n  RHO_INF: 1.0\n' "$TMP/rho1.4C.yaml"
deck ""                  $'STRUCTURAL DYNAMIC/GENALPHA:\n  RHO_INF: 0.5\n' "$TMP/rho05.4C.yaml"
deck $'\n  RHO_INF: 0.5' ""                                                "$TMP/wrong.4C.yaml"

probe DEFAULT       "$TMP/default.4C.yaml"
probe RHO_ONE       "$TMP/rho1.4C.yaml"
probe RHO_HALF      "$TMP/rho05.4C.yaml"
probe WRONG_SECTION "$TMP/wrong.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/DEFAULT.log"
# RHO_INF 1.0 written out explicitly reproduces the default bit for bit.
echo "RHO_ONE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/RHO_ONE.log")"
# The derived generalised-alpha coefficients 4C prints for each arm.
echo "DEFAULT_COEFFS=$(grep -oE '^   (rho|beta|gamma|alpha_f|alpha_m) = [0-9.]+' "$TMP/DEFAULT.log" | tr -d ' ' | tr '\n' ' ')"
echo "RHO_HALF_COEFFS=$(grep -oE '^   (rho|beta|gamma|alpha_f|alpha_m) = [0-9.]+' "$TMP/RHO_HALF.log" | tr -d ' ' | tr '\n' ' ')"
# ...and the answer moves with it.
grep -m1 -oE "is WRONG --> actresult=[^,]*" "$TMP/RHO_HALF.log"
# Wrong section: rejected at parse, with the offending key echoed.
grep -m1 -F "Could not match this input" "$TMP/WRONG_SECTION.log"
grep -m1 -F "RHO_INF: 0.5" "$TMP/WRONG_SECTION.log"
exit 0
