#!/bin/bash
# Tier-2 for fourc::input_format#19 — RESULT DESCRIPTION only tests
# abs(actresult - VALUE) > TOLERANCE, so a wide TOLERANCE turns the self-check
# into a rubber stamp: VALUE 0.0 against a true answer of 4.5e-03 is reported
# "is CORRECT" and the run exits 0.
#
# The two ways of getting the tolerance wrong that DO fail are pinned as well,
# because they bound the failure mode: TOLERANCE <= 0 is rejected at runtime by
# the result-test base class, and omitting TOLERANCE is a parse error.  What is
# left — a tolerance that is merely far too big — is the only silent one.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = RESULT DESCRIPTION tail, $2 = out file
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
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
$1
YAML
}

# The true answer, pinned tightly.  This is the reference the other arms lie about.
mk '      VALUE: 4.47909266337460053e-03
      TOLERANCE: 1e-12' "$TMP/tight.yaml"
# A deliberately absurd claim (zero) waved through by an absurd tolerance.
mk '      VALUE: 0.0
      TOLERANCE: 1.0' "$TMP/loose.yaml"
# TOLERANCE 0 is rejected outright...
mk '      VALUE: 4.47909266337460053e-03
      TOLERANCE: 0.0' "$TMP/zero.yaml"
# ...and omitting it never reaches the solver at all.
mk '      VALUE: 4.47909266337460053e-03' "$TMP/none.yaml"

probe TIGHT "$TMP/tight.yaml"
probe LOOSE "$TMP/loose.yaml"
probe ZERO  "$TMP/zero.yaml"
probe NONE  "$TMP/none.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/TIGHT.log"
# The rubber stamp, verbatim: a claim of 0.0 against 4.5e-03 is "CORRECT".
grep -m1 -F "is CORRECT, abs(diff)= 4.47909266337460053e-03 < 1.00000000000000000e+00" "$TMP/LOOSE.log"
# abs(diff) is the whole quantity, i.e. the check constrained nothing.
echo "LOOSE_DIFF_IS_THE_WHOLE_ANSWER=$(grep -c 'abs(diff)= 4.47909266337460053e-03' "$TMP/LOOSE.log")"
echo "LOOSE_REPORTED_WRONG=$(grep -c 'is WRONG' "$TMP/LOOSE.log")"
grep -m1 -F "Tolerance for result test must be strictly positive!" "$TMP/ZERO.log"
grep -m1 -F "4C_utils_result_test.cpp" "$TMP/ZERO.log"
grep -m1 -F "Could not match this input" "$TMP/NONE.log"
grep -m1 -F "[X] Expected parameter 'TOLERANCE'" "$TMP/NONE.log"
exit 0
