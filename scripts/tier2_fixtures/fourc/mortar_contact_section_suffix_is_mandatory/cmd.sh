#!/bin/bash
# Tier-2 for fourc::contact#13 — the dimension suffix is part of the section
# name.  There are exactly two spellings and nothing else is accepted.
#
#   DESIGN SURF MORTAR CONTACT CONDITIONS 3D   (3D decks)
#   DESIGN LINE MORTAR CONTACT CONDITIONS 2D   (2D decks)
#
# Six arms: a working 3D deck and a working 2D deck, each also run with the
# suffix dropped and with the other dimension's suffix.  Every wrong spelling is
# caught before anything runs, by name, with
#
#   Section '<what you wrote>' is not a valid section name.
#
# Note the check is on the SECTION NAME only: 'DESIGN SURF ... 2D' and
# 'DESIGN LINE ... 3D' are rejected even though both halves exist somewhere in
# the grammar, and the 3D deck is otherwise untouched.
. "$(dirname "$0")/../_lib/preamble.sh"

deck3d() {  # $1 = section name, $2 = out
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
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
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, -0.3]
    FUNCT: [0, 0, 1]
$1:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
  - "NODE 8 DSURFACE 2"
  - "NODE 9 DSURFACE 3"
  - "NODE 10 DSURFACE 3"
  - "NODE 11 DSURFACE 3"
  - "NODE 12 DSURFACE 3"
  - "NODE 13 DSURFACE 4"
  - "NODE 14 DSURFACE 4"
  - "NODE 15 DSURFACE 4"
  - "NODE 16 DSURFACE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
  - "NODE 9 COORD 0.0 0.0 1.1"
  - "NODE 10 COORD 1.0 0.0 1.1"
  - "NODE 11 COORD 1.0 1.0 1.1"
  - "NODE 12 COORD 0.0 1.0 1.1"
  - "NODE 13 COORD 0.0 0.0 2.1"
  - "NODE 14 COORD 1.0 0.0 2.1"
  - "NODE 15 COORD 1.0 1.0 2.1"
  - "NODE 16 COORD 0.0 1.0 2.1"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
  - "2 SOLID HEX8 9 10 11 12 13 14 15 16 MAT 1 KINEM nonlinear"
YAML
}

deck2d() {  # $1 = section name, $2 = out
cat > "$2" <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
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
  - E: 4
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, -0.3]
    FUNCT: [0, 1]
$1:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 2 DLINE 1"
  - "NODE 3 DLINE 2"
  - "NODE 4 DLINE 2"
  - "NODE 5 DLINE 3"
  - "NODE 6 DLINE 3"
  - "NODE 7 DLINE 4"
  - "NODE 8 DLINE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 1.1 0.0"
  - "NODE 6 COORD 1.0 1.1 0.0"
  - "NODE 7 COORD 1.0 2.1 0.0"
  - "NODE 8 COORD 0.0 2.1 0.0"
STRUCTURE ELEMENTS:
  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
  - "2 WALL QUAD4 5 6 7 8 MAT 1 KINEM nonlinear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
YAML
}

deck3d "DESIGN SURF MORTAR CONTACT CONDITIONS 3D" "$TMP/s3d.yaml"
deck3d "DESIGN SURF MORTAR CONTACT CONDITIONS"    "$TMP/s_none.yaml"
deck3d "DESIGN SURF MORTAR CONTACT CONDITIONS 2D" "$TMP/s2d.yaml"
deck2d "DESIGN LINE MORTAR CONTACT CONDITIONS 2D" "$TMP/l2d.yaml"
deck2d "DESIGN LINE MORTAR CONTACT CONDITIONS"    "$TMP/l_none.yaml"
deck2d "DESIGN LINE MORTAR CONTACT CONDITIONS 3D" "$TMP/l3d.yaml"

probe SURF_3D      "$TMP/s3d.yaml"
probe SURF_NOSUFIX "$TMP/s_none.yaml"
probe SURF_2D      "$TMP/s2d.yaml"
probe LINE_2D      "$TMP/l2d.yaml"
probe LINE_NOSUFIX "$TMP/l_none.yaml"
probe LINE_3D      "$TMP/l3d.yaml"

# Both correct spellings run to completion.
grep -m1 -F "processor 0 finished normally" "$TMP/SURF_3D.log"
grep -m1 -F "processor 0 finished normally" "$TMP/LINE_2D.log"
echo "STEPS_SURF_3D=$(grep -c 'Finalised step' "$TMP/SURF_3D.log")"
echo "STEPS_LINE_2D=$(grep -c 'Finalised step' "$TMP/LINE_2D.log")"

# Every wrong spelling is named back at you and nothing runs.
grep -m1 -F "Section 'DESIGN SURF MORTAR CONTACT CONDITIONS' is not a valid section name." "$TMP/SURF_NOSUFIX.log"
grep -m1 -F "Section 'DESIGN SURF MORTAR CONTACT CONDITIONS 2D' is not a valid section name." "$TMP/SURF_2D.log"
grep -m1 -F "Section 'DESIGN LINE MORTAR CONTACT CONDITIONS' is not a valid section name." "$TMP/LINE_NOSUFIX.log"
grep -m1 -F "Section 'DESIGN LINE MORTAR CONTACT CONDITIONS 3D' is not a valid section name." "$TMP/LINE_3D.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/SURF_NOSUFIX.log"
for a in SURF_NOSUFIX SURF_2D LINE_NOSUFIX LINE_3D; do
  echo "REACHED_FILL_COMPLETE_$a=$(grep -c 'fill_complete() on discretization structure' "$TMP/$a.log")"
done
exit 0
