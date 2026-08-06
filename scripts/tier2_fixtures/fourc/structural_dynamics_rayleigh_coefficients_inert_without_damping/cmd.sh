#!/bin/bash
# Tier-2 for fourc::structural_dynamics#4 — M_DAMP and K_DAMP do nothing on
# their own.  They are read only when DAMPING is switched to "Rayleigh", whose
# default is "None", and a deck that sets the two coefficients and forgets the
# switch runs to completion with the UNDAMPED answer and no warning.
#
#   UNDAMPED           nothing set                     -> exit 0 (reference)
#   COEFFS_ONLY        M_DAMP + K_DAMP, DAMPING default -> exit 0, SAME answer
#                                                          to the pinned 1e-9
#   RAYLEIGH           DAMPING Rayleigh + coefficients  -> different answer
#   RAYLEIGH_NO_COEFFS DAMPING Rayleigh, no coefficients-> hard abort naming
#                                                          the missing key
# The last arm is the one piece of loudness in the whole area: 4C refuses to
# guess a coefficient, but it will happily ignore one you did supply.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra STRUCTURAL DYNAMIC lines, $2 = out
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:$1
  DYNAMICTYPE: "GenAlpha"
  TIMESTEP: 0.05
  NUMSTEP: 8
  MAXTIME: 0.4
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
      VALUE: 1.70856951294205382e-02
      TOLERANCE: 1.0e-09
YAML
}

deck ""                                                              "$TMP/undamped.4C.yaml"
deck $'\n  M_DAMP: 5.0\n  K_DAMP: 0.05'                              "$TMP/coeffs.4C.yaml"
deck $'\n  DAMPING: "Rayleigh"\n  M_DAMP: 5.0\n  K_DAMP: 0.05'       "$TMP/rayleigh.4C.yaml"
deck $'\n  DAMPING: "Rayleigh"'                                      "$TMP/nocoeffs.4C.yaml"

probe UNDAMPED           "$TMP/undamped.4C.yaml"
probe COEFFS_ONLY        "$TMP/coeffs.4C.yaml"
probe RAYLEIGH           "$TMP/rayleigh.4C.yaml"
probe RAYLEIGH_NO_COEFFS "$TMP/nocoeffs.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/UNDAMPED.log"
# The coefficients alone change nothing: same result test, still passes.
echo "COEFFS_ONLY_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/COEFFS_ONLY.log")"
echo "COEFFS_ONLY_WARNINGS=$(grep -ciE 'M_DAMP|K_DAMP|damping' "$TMP/COEFFS_ONLY.log")"
# With DAMPING switched on, the same coefficients bite.
grep -m1 -oE "is WRONG --> actresult=[^,]*" "$TMP/RAYLEIGH.log"
# And DAMPING without coefficients is a hard error that names the key.
grep -m1 -F "Rayleigh damping parameter K_DAMP not explicitly given." "$TMP/RAYLEIGH_NO_COEFFS.log"
grep -m1 -F "4C_adapter_str_structure_new.cpp" "$TMP/RAYLEIGH_NO_COEFFS.log"
exit 0
