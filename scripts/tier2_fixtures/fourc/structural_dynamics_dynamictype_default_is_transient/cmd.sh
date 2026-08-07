#!/bin/bash
# Tier-2 for fourc::structural_dynamics#10 — DYNAMICTYPE is OPTIONAL and its
# default is GenAlpha, i.e. a TRANSIENT scheme.  Omitting it does not give you a
# static analysis and does not warn: the deck parses, exits 0 and returns a
# different number, because inertia and the generalised-alpha averaging are now
# in the residual.
#
#   STATICS           DYNAMICTYPE: "Statics" -> exit 0, pinned answer
#   NO_DYNAMICTYPE    line deleted           -> exit 0 as far as the solver is
#                                               concerned, but a different
#                                               answer and the GenAlpha
#                                               coefficient banner in the log
#   NO_LINEAR_SOLVER  LINEAR_SOLVER deleted  -> the opposite case: its default
#                                               of -1 is not a valid solver id,
#                                               so it fails loudly at once
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = DYNAMICTYPE line (may be empty), $2 = LINEAR_SOLVER line, $3 = out
cat > "$3" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:$1
  TIMESTEP: 0.25
  NUMSTEP: 4
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-08
  MAXITER: 30$2
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

deck $'\n  DYNAMICTYPE: "Statics"' $'\n  LINEAR_SOLVER: 1' "$TMP/statics.4C.yaml"
deck ""                            $'\n  LINEAR_SOLVER: 1' "$TMP/nodyn.4C.yaml"
deck $'\n  DYNAMICTYPE: "Statics"' ""                      "$TMP/nosolver.4C.yaml"

probe STATICS          "$TMP/statics.4C.yaml"
probe NO_DYNAMICTYPE   "$TMP/nodyn.4C.yaml"
probe NO_LINEAR_SOLVER "$TMP/nosolver.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/STATICS.log"
# Omitting DYNAMICTYPE quietly selects a transient scheme...
echo "NO_DYNAMICTYPE_STEPS=$(grep -c '^Finalised step' "$TMP/NO_DYNAMICTYPE.log")"
echo "NO_DYNAMICTYPE_GENALPHA_BANNER=$(grep -cE '^   alpha_m = ' "$TMP/NO_DYNAMICTYPE.log")"
echo "NO_DYNAMICTYPE_MENTIONS_THE_KEY=$(grep -ci 'dynamictype' "$TMP/NO_DYNAMICTYPE.log")"
# ...and returns a different number for the same deck.
grep -m1 -oE "is WRONG --> actresult=[^,]*" "$TMP/NO_DYNAMICTYPE.log"
# LINEAR_SOLVER is the opposite case: omitting it fails immediately and by name.
grep -m1 -F "no linear solver defined for structural field. Please set LINEAR_SOLVER in STRUCTURAL DYNAMIC to a valid number!" "$TMP/NO_LINEAR_SOLVER.log"
grep -m1 -F "4C_structure_new_solver_factory.cpp" "$TMP/NO_LINEAR_SOLVER.log"
exit 0
