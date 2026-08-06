#!/bin/bash
# Tier-2 for fourc::solid_mechanics#6 — INT_STRATEGY: "Old" is not a cosmetic
# compatibility switch.  On a plain structural deck it silently replaces the
# whole nonlinear solver stack: the NOX status-test machinery disappears and the
# legacy integrator takes over.  The ANSWER is the same to the last bit, so
# nothing in the numbers warns you; only the shape of the log changes.
#
#   STANDARD  default        -> NOX: "=== Structural predictor: ... ===",
#                              "-- Status Test Results --", "nlniter" banner
#   OLD       INT_STRATEGY   -> legacy: "with statics", "numiter" table,
#             = "Old"           and ZERO NOX status-test blocks
#
# The loud sibling is what happens the moment the structural field is coupled:
# an SSI problem asserts on it by name.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra STRUCTURAL DYNAMIC line, $2 = out
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:$1
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
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
      VALUE: 4.45356052073027156e-02
      TOLERANCE: 1.0e-10
YAML
}

deck ""                        "$TMP/standard.4C.yaml"
deck $'\n  INT_STRATEGY: "Old"' "$TMP/old.4C.yaml"

probe STANDARD "$TMP/standard.4C.yaml"
probe OLD      "$TMP/old.4C.yaml"

# Both reach the same answer: the pinned result test passes in both arms.
echo "OLD_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/OLD.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/OLD.log"
# But the solver stack underneath is a different one.
echo "STANDARD_NOX_STATUS_BLOCKS=$(grep -c -- '-- Status Test Results --' "$TMP/STANDARD.log")"
echo "OLD_NOX_STATUS_BLOCKS=$(grep -c -- '-- Status Test Results --' "$TMP/OLD.log")"
grep -m1 -oE "^Finalised step 1 / 1 .*nlniter [0-9]+" "$TMP/STANDARD.log"
grep -m1 -oE "^Finalised step 1 / 1 .*numiter [0-9]+" "$TMP/OLD.log"
grep -m1 -F "Structural predictor for field 'structure' ConstDis yields absolute res-norm" "$TMP/OLD.log"

# Loud sibling: the same switch on a coupled problem is a hard assert.
SSI=$(upstream ssi_2D_quad4.4C.yaml) || exit 3
grep -q "^STRUCTURAL DYNAMIC:" "$SSI" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
python3 - "$SSI" "$TMP/ssi_old.4C.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
t = t.replace("STRUCTURAL DYNAMIC:", 'STRUCTURAL DYNAMIC:\n  INT_STRATEGY: "Old"', 1)
open(sys.argv[2], "w").write(t)
PY
probe SSI_OLD "$TMP/ssi_old.4C.yaml"
grep -m1 -F 'Only the new solid time integration is supported for SSI problems. Set `INT_STRATEGY` to `Standard`!' "$TMP/SSI_OLD.log"
grep -m1 -F "4C_ssi_dyn.cpp" "$TMP/SSI_OLD.log"
exit 0
