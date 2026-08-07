#!/bin/bash
# Tier-2 for fourc::solid_mechanics#5 — and a FALSIFICATION of it.
#
# Claimed: structural Neumann conditions take NUMDOF: 6 (3 forces + 3 moments),
#          and NUMDOF: 3 "silently drops the moment components".
# Observed: on a 3D SOLID continuum mesh a node has THREE dofs.
#   NUMDOF: 3                     -> runs, exit 0
#   NUMDOF: 6 with the moment slots zeroed
#                                 -> runs, exit 0, SAME answer to the last bit
#   NUMDOF: 6 with a moment slot switched ON
#                                 -> ABORTS LOUDLY at the first Neumann
#                                    evaluation, exit 1
# So nothing is dropped silently; the extra slots are simply not usable on a
# continuum element, and switching one on is a hard error.  Moment loads belong
# on elements that carry rotational dofs (SHELL7P uses NUMDOF: 6, BEAM3R uses
# NUMDOF: 9) — not on SOLID.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = NUMDOF, $2 = ONOFF list, $3 = VAL list, $4 = FUNCT list, $5 = out
cat > "$5" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
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
    NUMDOF: $1
    ONOFF: $2
    VAL: $3
    FUNCT: $4
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
      TOLERANCE: 1.0e-12
YAML
}

deck 3 "[0, 1, 0]"             "[0.0, 10.0, 0.0]"                  "[0, 1, 0]"             "$TMP/n3.4C.yaml"
deck 6 "[0, 1, 0, 0, 0, 0]"    "[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]"   "[0, 1, 0, 0, 0, 0]"    "$TMP/n6.4C.yaml"
deck 6 "[0, 1, 0, 0, 1, 0]"    "[0.0, 10.0, 0.0, 0.0, 10.0, 0.0]"  "[0, 1, 0, 0, 1, 0]"    "$TMP/n6m.4C.yaml"

probe NUMDOF3          "$TMP/n3.4C.yaml"
probe NUMDOF6_ZEROED   "$TMP/n6.4C.yaml"
probe NUMDOF6_MOMENT   "$TMP/n6m.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/NUMDOF3.log"
# Same result test, pinned to 1e-12, passes for both NUMDOF spellings.
echo "NUMDOF6_ZEROED_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NUMDOF6_ZEROED.log")"
# Switching a moment slot on is a hard error, not a silent drop.
grep -m1 -F "Number of Dimensions in Neumann_Evaluation is 3. Further DoFs are not considered." "$TMP/NUMDOF6_MOMENT.log"
grep -m1 -F "4C_solid_3D_ele_surface_evaluate.cpp" "$TMP/NUMDOF6_MOMENT.log"
exit 0
