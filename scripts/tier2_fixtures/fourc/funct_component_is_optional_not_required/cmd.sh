#!/bin/bash
# Tier-2 for fourc::input_format#1 — a FALSIFICATION.
#
# Claimed: SYMBOLIC_FUNCTION_OF_SPACE_TIME with a VARIABLE REQUIRES
#          'COMPONENT: 0' in the same list item; omitting it silently ignores
#          the VARIABLE and the function evaluates to 0 everywhere, so a
#          Dirichlet driven by it stays stuck at 0 instead of ramping.
#
# Observed: COMPONENT is declared as an OPTIONAL parameter and defaults to 0
#          (core/utils/src/functions/4C_utils_function.cpp uses
#          .value_or(0)).  Omitting it changes nothing: the VARIABLE is read,
#          the function returns its value, and the run produces a displacement
#          BIT-IDENTICAL to the arm that spells COMPONENT out.  Nothing is
#          stuck at 0.
#
# What DOES fail is giving a component index that is not the one being defined:
#          "expected COMPONENT 0 but got COMPONENT 1"  (4C_utils_function.cpp).
#
# The probe is built so the two failure modes cannot be confused with success:
# the variable ramps to 0.5, not to 1.0, and the prescribed displacement is 0.1,
# so the correct answer is 0.05.  A silently ignored VARIABLE would give 0.0
# (function 0) or 0.1 (function 1) -- neither of which is what 4C prints.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = first line of the FUNCT1 list item(s), $2 = out file
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.25
  NUMSTEP: 4
  MAXTIME: 1.0
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
FUNCT1:
$1
  - VARIABLE: 0
    NAME: "a"
    TYPE: "linearinterpolation"
    NUMPOINTS: 2
    TIMES: [0, 1]
    VALUES: [0, 0.5]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 0.1, 0]
    FUNCT: [0, 1, 0]
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
      VALUE: 5.00000000000000028e-02
      TOLERANCE: 1e-12
YAML
}

mk '  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "a"' "$TMP/with.yaml"
mk '  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "a"' "$TMP/without.yaml"
mk '  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "a"' "$TMP/comp1.yaml"

# Pin that the second deck really does omit the key, so this cannot quietly
# become a test of two identical inputs.
echo "WITHOUT_DECK_MENTIONS_COMPONENT=$(grep -c COMPONENT "$TMP/without.yaml")"
echo "WITH_DECK_MENTIONS_COMPONENT=$(grep -c COMPONENT "$TMP/with.yaml")"

probe WITH    "$TMP/with.yaml"
probe WITHOUT "$TMP/without.yaml"
probe COMP1   "$TMP/comp1.yaml"

# Both spellings hit the pinned answer 0.05 = VAL 0.1 x variable 0.5 exactly.
grep -m1 -F "is CORRECT, abs(diff)= 0.00000000000000000e+00" "$TMP/WITH.log"
grep -m1 -F "is CORRECT, abs(diff)= 0.00000000000000000e+00" "$TMP/WITHOUT.log"
grep -m1 -F "processor 0 finished normally" "$TMP/WITHOUT.log"
# The claimed symptom -- a BC stuck at 0 because the VARIABLE was dropped --
# does not occur: the arm without COMPONENT reports no failing test at all.
echo "WITHOUT_REPORTED_A_WRONG_TEST=$(grep -c 'is WRONG' "$TMP/WITHOUT.log")"
echo "WITHOUT_COMPLAINED_ABOUT_COMPONENT=$(grep -ci 'COMPONENT' "$TMP/WITHOUT.log")"
# What actually fails is a MISMATCHED component index.
grep -m1 -F "expected COMPONENT 0 but got COMPONENT 1" "$TMP/COMP1.log"
grep -m1 -F "4C_utils_function.cpp" "$TMP/COMP1.log"

# The schema itself declares COMPONENT optional.
"$BIN" --parameters 2>/dev/null > "$TMP/params.yaml"
echo "SCHEMA_COMPONENT_REQUIRED_TRUE=$(grep -A2 'name: COMPONENT$' "$TMP/params.yaml" | grep -c 'required: true')"
exit 0
