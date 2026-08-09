#!/bin/bash
# Tier-2 for fourc::structural_mechanics#11 — DENS is a REQUIRED key of
# MAT_Struct_StVenantKirchhoff in every analysis type, Statics included.
#
# The advice this entry corrects ("density is only needed for dynamics") is
# about whether the VALUE matters.  Both halves are checked here:
#
#   Statics, DENS present            -> exit 0
#   Statics, DENS omitted            -> exit 1, Expected parameter 'DENS'
#   OneStepTheta, DENS omitted       -> exit 1, same message
#   Statics, DENS 1.0 vs DENS 1e+06  -> identical answer to 1e-12
#
# So the value really is irrelevant under Statics, and the key is still
# mandatory.  The spec lives in global_legacy_module_validmaterials and has no
# default, which is why the failure is a parse failure and not a warning.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 DYNAMICTYPE, $2 material body, $3 extra sections, $4 out
cat > "$4" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "$1"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-12
  TOLRES: 1.0e-11
  MAXITER: 30
  LINEAR_SOLVER: 1
${3}SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
$2
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
    VAL: [0.0, 1.0, 0.0]
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
      NODE: 6
      QUANTITY: "dispy"
      VALUE: 4.48471241704061219e-03
      TOLERANCE: 1.0e-12
YAML
}

WITH_DENS='      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0'
HUGE_DENS='      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1000000.0'
NO_DENS='      YOUNG: 1000.0
      NUE: 0.3'
OST='STRUCTURAL DYNAMIC/ONESTEPTHETA:
  THETA: 1.0
'

deck Statics      "$WITH_DENS" ""     "$TMP/statics_dens.yaml"
deck Statics      "$HUGE_DENS" ""     "$TMP/statics_huge.yaml"
deck Statics      "$NO_DENS"   ""     "$TMP/statics_nodens.yaml"
deck OneStepTheta "$NO_DENS"   "$OST" "$TMP/ost_nodens.yaml"

probe STATICS_DENS1    "$TMP/statics_dens.yaml"
probe STATICS_DENS1E6  "$TMP/statics_huge.yaml"
probe STATICS_NO_DENS  "$TMP/statics_nodens.yaml"
probe OST_NO_DENS      "$TMP/ost_nodens.yaml"

# The value is irrelevant under Statics: six orders of magnitude of density,
# the same pinned result test, no failures either way.
grep -m1 -F "processor 0 finished normally" "$TMP/STATICS_DENS1.log"
echo "STATICS_DENS1_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/STATICS_DENS1.log")"
echo "STATICS_DENS1E6_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/STATICS_DENS1E6.log")"

# The key is mandatory all the same, and the rejection is at parse.
grep -m1 -F "Failed to match specification in section 'MATERIALS'" "$TMP/STATICS_NO_DENS.log"
grep -m1 -F "Expected parameter 'DENS'" "$TMP/STATICS_NO_DENS.log"
grep -m1 -F "4C_global_data_read.cpp" "$TMP/STATICS_NO_DENS.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/STATICS_NO_DENS.log"
grep -m1 -F "Expected parameter 'DENS'" "$TMP/OST_NO_DENS.log"
echo "OST_NO_DENS_MISSING_KEY_REPORTS=$(grep -c "Expected parameter 'DENS'" "$TMP/OST_NO_DENS.log")"
echo "OST_NO_DENS_REACHED_FILL_COMPLETE=$(grep -c 'fill_complete() on discretization structure' "$TMP/OST_NO_DENS.log")"

# Nothing ever reaches the mesh in the two failing arms.
echo "STATICS_NO_DENS_REACHED_FILL_COMPLETE=$(grep -c 'fill_complete() on discretization structure' "$TMP/STATICS_NO_DENS.log")"
echo "STATICS_DENS1_REACHED_FILL_COMPLETE=$(grep -c 'fill_complete() on discretization structure' "$TMP/STATICS_DENS1.log")"
exit 0
