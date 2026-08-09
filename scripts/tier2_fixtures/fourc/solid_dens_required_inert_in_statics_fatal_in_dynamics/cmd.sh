#!/bin/bash
# Tier-2 for fourc::solid_mechanics#2 — and a FALSIFICATION of the half of that
# entry that said DENS "can be omitted" for quasi-static problems.
#
# DENS is a REQUIRED key of MAT_Struct_StVenantKirchhoff.  Leaving it out is a
# parse error, not a default.  What IS true is that its VALUE does not matter
# under Statics without gravity: DENS 0.0 and DENS 1.0 give the same static
# answer to the last bit, and the deck's own result test passes for both.  Under
# a transient scheme the same DENS 0.0 kills the run before step 1.
#
# Four arms on one single-HEX8 deck:
#   STATICS_DENS1  Statics,  DENS 1.0  -> exit 0, result test passes
#   STATICS_DENS0  Statics,  DENS 0.0  -> exit 0, SAME result test passes
#   NO_DENS        Statics,  key gone  -> parse error, exit 1
#   GENALPHA_DENS0 GenAlpha, DENS 0.0  -> singular-matrix throw, exit 1
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = DYNAMICTYPE, $2 = DENS line (may be empty), $3 = output file
cat > "$3" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "$1"
  TIMESTEP: 0.25
  NUMSTEP: 4
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
      YOUNG: 1000
      NUE: 0.3$2
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
    VAL: [0, 10, 0]
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

deck Statics  $'\n      DENS: 1.0' "$TMP/s1.4C.yaml"
deck Statics  $'\n      DENS: 0.0' "$TMP/s0.4C.yaml"
deck Statics  ""                   "$TMP/sx.4C.yaml"
deck GenAlpha $'\n      DENS: 0.0' "$TMP/g0.4C.yaml"

probe STATICS_DENS1  "$TMP/s1.4C.yaml"
probe STATICS_DENS0  "$TMP/s0.4C.yaml"
probe NO_DENS        "$TMP/sx.4C.yaml"
probe GENALPHA_DENS0 "$TMP/g0.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/STATICS_DENS1.log"
# Same static answer with the density zeroed: the result test still passes.
echo "STATICS_DENS0_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/STATICS_DENS0.log")"
# The key cannot simply be left out.
grep -m1 -F "Could not match this input" "$TMP/NO_DENS.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/NO_DENS.log"
# Under a transient scheme the same zero density is fatal at step 0.
grep -m1 -F "You are about to invert a singular matrix!" "$TMP/GENALPHA_DENS0.log"
grep -m1 -F "4C_structure_new_integrator.cpp" "$TMP/GENALPHA_DENS0.log"
echo "GENALPHA_DENS0_STEPS=$(grep -c '^Finalised step' "$TMP/GENALPHA_DENS0.log")"
exit 0
