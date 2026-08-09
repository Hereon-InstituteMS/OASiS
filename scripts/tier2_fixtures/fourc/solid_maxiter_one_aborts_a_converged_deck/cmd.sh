#!/bin/bash
# Tier-2 for fourc::solid_mechanics#1 — MAXITER: 1 is not "one iteration and
# stop", it is an ABORT.  4C credits the iteration counter before the
# convergence test, so a deck that converges in three Newton steps under
# MAXITER: 30 dies under MAXITER: 1 with a NOX status-test failure and exit 1.
#
# Same single-HEX8 deck twice; the only difference is the MAXITER value.
# The counter line 4C prints is the giveaway and is asserted literally.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = MAXITER value, $2 = output file
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: $1
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
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
    VAL: [0, 200, 0]
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
YAML
}

deck 30 "$TMP/many.4C.yaml"
deck 1  "$TMP/one.4C.yaml"

probe MAXITER30 "$TMP/many.4C.yaml"
probe MAXITER1  "$TMP/one.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/MAXITER30.log"
grep -oE "Number of Iterations = [0-9]+ < 30" "$TMP/MAXITER30.log" | tail -1
grep -m1 -oE "^Finalised step 1 / 1 .*nlniter [0-9]+" "$TMP/MAXITER30.log"
grep -m1 -F "Failed.......Number of Iterations = 1 < 1" "$TMP/MAXITER1.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/MAXITER1.log"
# The abort happens before any step is finalised at all.
echo "MAXITER1_STEPS_FINALISED=$(grep -c '^Finalised step' "$TMP/MAXITER1.log")"
exit 0
