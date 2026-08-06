#!/bin/bash
# Tier-2 for fourc::structural_mechanics#7 — what 4C checks on a body-force
# block is INTERNAL CONSISTENCY, not that NUMDOF matches the element.
#
# Seven arms.  DESIGN VOL NEUMANN on a 3D SOLID HEX8 and DESIGN SURF NEUMANN on
# a 2D WALL QUAD4, same material, same load:
#
#   3D NUMDOF 3                      -> runs, exit 0
#   3D NUMDOF 6, extra slots OFF     -> runs, exit 0, SAME answer to 1e-12
#   3D NUMDOF 6, a 4th slot ON       -> aborts (the continuum element has 3 dofs)
#   3D NUMDOF 6 with a 3-entry ONOFF -> rejected at parse
#   3D NUMDOF 3 with a 6-entry ONOFF -> rejected at parse
#   2D NUMDOF 2                      -> runs, exit 0
#   2D NUMDOF 3, the 3rd slot ON     -> runs, exit 0, SAME answer, NO diagnostic
#
# The last pair is the trap the entry is about: an oversized but self-consistent
# block is not caught, so a wrong component count is a silent modelling error.
# The string an earlier version quoted, 'NUMDOF mismatch - expected 3 got 6', is
# in no log and CLAIMED_NUMDOF_MISMATCH_TEXT=0 keeps that pinned.
. "$(dirname "$0")/../_lib/preamble.sh"

deck3d() {  # $1 NUMDOF, $2 ONOFF, $3 VAL, $4 FUNCT, $5 out
cat > "$5" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-12
  TOLRES: 1.0e-11
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
DESIGN VOL NEUMANN CONDITIONS:
  - E: 1
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
DVOL-NODE TOPOLOGY:
  - "NODE 1 DVOL 1"
  - "NODE 2 DVOL 1"
  - "NODE 3 DVOL 1"
  - "NODE 4 DVOL 1"
  - "NODE 5 DVOL 1"
  - "NODE 6 DVOL 1"
  - "NODE 7 DVOL 1"
  - "NODE 8 DVOL 1"
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
      NODE: 6
      QUANTITY: "dispy"
      VALUE: 2.30969996560593610e-01
      TOLERANCE: 1.0e-12
YAML
}

deck2d() {  # $1 NUMDOF, $2 ONOFF, $3 VAL, $4 FUNCT, $5 out
cat > "$5" <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-12
  TOLRES: 1.0e-11
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
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: $1
    ONOFF: $2
    VAL: $3
    FUNCT: $4
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 2
      QUANTITY: "dispy"
      VALUE: 2.22460998093427315e-01
      TOLERANCE: 1.0e-12
YAML
}

deck3d 3 "[0, 1, 0]"          "[0.0, 100.0, 0.0]"                   "[0, 1, 0]"          "$TMP/v3.yaml"
deck3d 6 "[0, 1, 0, 0, 0, 0]" "[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]"    "[0, 1, 0, 0, 0, 0]" "$TMP/v6z.yaml"
deck3d 6 "[0, 1, 0, 0, 1, 0]" "[0.0, 100.0, 0.0, 0.0, 100.0, 0.0]"  "[0, 1, 0, 0, 1, 0]" "$TMP/v6on.yaml"
deck3d 6 "[0, 1, 0]"          "[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]"    "[0, 1, 0, 0, 0, 0]" "$TMP/vshort.yaml"
deck3d 3 "[0, 1, 0, 0, 0, 0]" "[0.0, 100.0, 0.0]"                   "[0, 1, 0]"          "$TMP/vlong.yaml"
deck2d 2 "[0, 1]"             "[0.0, 100.0]"                        "[0, 1]"             "$TMP/w2.yaml"
deck2d 3 "[0, 1, 1]"          "[0.0, 100.0, 100.0]"                 "[0, 1, 1]"          "$TMP/w3on.yaml"

probe VOL_NUMDOF3     "$TMP/v3.yaml"
probe VOL_NUMDOF6_OFF "$TMP/v6z.yaml"
probe VOL_NUMDOF6_ON  "$TMP/v6on.yaml"
probe VOL_ONOFF_SHORT "$TMP/vshort.yaml"
probe VOL_ONOFF_LONG  "$TMP/vlong.yaml"
probe WALL_NUMDOF2    "$TMP/w2.yaml"
probe WALL_NUMDOF3_ON "$TMP/w3on.yaml"

# The correct 3D body force runs, and the oversized-but-consistent one gives the
# same answer to the last bit — the same pinned RESULT DESCRIPTION passes.
grep -m1 -F "processor 0 finished normally" "$TMP/VOL_NUMDOF3.log"
echo "VOL_NUMDOF3_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/VOL_NUMDOF3.log")"
echo "VOL_NUMDOF6_OFF_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/VOL_NUMDOF6_OFF.log")"

# Switching a fourth slot ON is caught on a 3D continuum element.
grep -m1 -F "You have activated more than 3 dofs in your Neumann boundary condition." "$TMP/VOL_NUMDOF6_ON.log"
grep -m1 -F "4C_solid_3D_ele_neumann_evaluator.cpp" "$TMP/VOL_NUMDOF6_ON.log"

# Internal inconsistency is what the parser really enforces, in both directions.
grep -m1 -F "Could not match this input" "$TMP/VOL_ONOFF_SHORT.log"
grep -m1 -F "4C_fem_condition_definition.cpp" "$TMP/VOL_ONOFF_SHORT.log"
grep -m1 -F "Could not match this input" "$TMP/VOL_ONOFF_LONG.log"

# On the 2D WALL element the extra component is not caught at all: same answer,
# no diagnostic, exit 0.
echo "WALL_NUMDOF2_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WALL_NUMDOF2.log")"
echo "WALL_NUMDOF3_ON_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WALL_NUMDOF3_ON.log")"
echo "WALL_NUMDOF3_ON_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/WALL_NUMDOF3_ON.log")"

python3 - "$TMP"/*.log <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += t.count("numdof mismatch")
print("CLAIMED_NUMDOF_MISMATCH_TEXT=%d" % n)
PY
exit 0
