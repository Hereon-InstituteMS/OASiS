#!/bin/bash
# Tier-2 for fourc::structural_dynamics#0 — DENS = 0 in a transient structural
# run, and what actually happens to each family of time integrator.
#
# The entry said an IMPLICIT scheme would lose its inertia term and "effectively
# become quasi-static while the user expected transient".  It does not.  Both
# families die at step 0 with the SAME linear-algebra message, and neither
# produces the quasi-static answer that the Statics control arm produces:
#
#   STATICS_REF     Statics,       DENS 1.0 -> exit 0, 4/4 steps
#   GENALPHA_DENS0  GenAlpha,      DENS 0.0 -> exit 1, 0 steps, singular matrix
#   EXPLICIT_DENS0  ExplicitEuler, DENS 0.0 -> exit 1, 0 steps, singular matrix
#
# And the diagnostic never says "density": grepping it for the word finds
# nothing, which is exactly why the entry warns you to check DENS by inspection.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = DYNAMICTYPE, $2 = TIMESTEP, $3 = DENS, $4 = out
cat > "$4" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "$1"
  TIMESTEP: $2
  NUMSTEP: 4
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-08
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
      DENS: $3
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
YAML
}

deck Statics       0.25   1.0 "$TMP/ref.4C.yaml"
deck GenAlpha      0.25   0.0 "$TMP/ga0.4C.yaml"
deck ExplicitEuler 1.0e-4 0.0 "$TMP/ex0.4C.yaml"

probe STATICS_REF    "$TMP/ref.4C.yaml"
probe GENALPHA_DENS0 "$TMP/ga0.4C.yaml"
probe EXPLICIT_DENS0 "$TMP/ex0.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/STATICS_REF.log"
echo "STATICS_REF_STEPS=$(grep -c '^Finalised step' "$TMP/STATICS_REF.log")"
echo "GENALPHA_DENS0_STEPS=$(grep -c '^Finalised step' "$TMP/GENALPHA_DENS0.log")"
echo "EXPLICIT_DENS0_STEPS=$(grep -c '^Finalised step' "$TMP/EXPLICIT_DENS0.log")"
# Same message from the same place for both integrator families.
grep -m1 -F "You are about to invert a singular matrix!" "$TMP/GENALPHA_DENS0.log"
grep -m1 -F "You are about to invert a singular matrix!" "$TMP/EXPLICIT_DENS0.log"
grep -m1 -F "4C_structure_new_integrator.cpp" "$TMP/GENALPHA_DENS0.log"
# ...and it never names the material parameter that caused it.
echo "DENSITY_WORD_IN_IMPLICIT_DIAGNOSTIC=$(grep -ci 'densit' "$TMP/GENALPHA_DENS0.log")"
echo "DENSITY_WORD_IN_EXPLICIT_DIAGNOSTIC=$(grep -ci 'densit' "$TMP/EXPLICIT_DENS0.log")"
exit 0
