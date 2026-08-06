#!/bin/bash
# Tier-2 for fourc::input_format#23 — the two DIFFERENT checks that NUMDOF has
# to pass, and the fact that Neumann is lenient in one direction while Dirichlet
# is not lenient at all.
#
#   1. ONOFF / VAL / FUNCT must have exactly NUMDOF entries.  This is a SPEC
#      check, so it fails at parse time with "Candidate parameter 'ONOFF' has
#      incorrect size" (and the same for VAL and FUNCT) inside the match tree.
#   2. NUMDOF itself must equal the nodal DOF count -- but only for Dirichlet.
#      A self-consistent NUMDOF: 2 block passes the spec and then throws from
#      the DBC extractor: "2 DOFs given but 3 expected in Surface Dirichlet
#      boundary condition".
#   3. A Neumann block may declare MORE entries than the element has DOFs -- the
#      classic NUMDOF: 6 structural convention -- and the surplus is ignored:
#      the answer is bit-identical to NUMDOF: 3.  Declaring FEWER is fatal, from
#      a third place again: solid_3D_ele_surface_evaluate.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = DIRICH block body, $2 = NEUMANN block body, $3 = out file
cat > "$3" <<YAML
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
$1
DESIGN SURF NEUMANN CONDITIONS:
$2
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
      VALUE: 4.47909266337460053e-03
      TOLERANCE: 1e-12
YAML
}

D3='  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]'
D3_SHORT_ARRAYS='  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]'
D2='  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]'
N3='  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 1, 0]
    FUNCT: [0, 0, 0]
    TYPE: "Live"'
N6='  - E: 2
    NUMDOF: 6
    ONOFF: [0, 1, 0, 0, 0, 0]
    VAL: [0, 1, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
    TYPE: "Live"'
N2='  - E: 2
    NUMDOF: 2
    ONOFF: [0, 1]
    VAL: [0, 1]
    FUNCT: [0, 0]
    TYPE: "Live"'

mk "$D3"               "$N3" "$TMP/good.yaml"
mk "$D3_SHORT_ARRAYS"  "$N3" "$TMP/short.yaml"
mk "$D2"               "$N3" "$TMP/dnumdof2.yaml"
mk "$D3"               "$N6" "$TMP/nnumdof6.yaml"
mk "$D3"               "$N2" "$TMP/nnumdof2.yaml"

probe GOOD  "$TMP/good.yaml"
probe SHORT "$TMP/short.yaml"
probe DIRI2 "$TMP/dnumdof2.yaml"
probe NEU6  "$TMP/nnumdof6.yaml"
probe NEU2  "$TMP/nnumdof2.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
# (1) arrays shorter than NUMDOF: a spec failure, all three arrays named.
grep -m1 -F "Candidate parameter 'ONOFF' has incorrect size" "$TMP/SHORT.log"
grep -m1 -F "Candidate parameter 'VAL' has incorrect size" "$TMP/SHORT.log"
grep -m1 -F "Candidate parameter 'FUNCT' has incorrect size" "$TMP/SHORT.log"
grep -m1 -F "Failed to match condition specification in section 'DESIGN SURF DIRICH CONDITIONS'" "$TMP/SHORT.log"
# (2) self-consistent but too small: passes the spec, dies in the DBC extractor.
grep -m1 -F "2 DOFs given but 3 expected in Surface Dirichlet boundary condition" "$TMP/DIRI2.log"
grep -m1 -F "4C_fem_discretization_utils_dbc.cpp" "$TMP/DIRI2.log"
echo "DIRI2_FAILED_THE_SPEC=$(grep -c 'has incorrect size' "$TMP/DIRI2.log")"
# (3) Neumann tolerates a surplus and reproduces the NUMDOF: 3 answer exactly.
grep -m1 -F "is CORRECT, abs(diff)= 0.00000000000000000e+00" "$TMP/NEU6.log"
# ...but not a shortfall, and the message comes from the element surface, not the DBC code.
grep -m1 -F "Fewer functions or curves defined than the element has dofs." "$TMP/NEU2.log"
grep -m1 -F "4C_solid_3D_ele_surface_evaluate.cpp" "$TMP/NEU2.log"
exit 0
