#!/bin/bash
# Tier-2 for fourc::input_format#24 — a DESIGN ... CONDITIONS entity id is
# 1-based in the input and must exist in the matching D<X>-NODE TOPOLOGY
# section.  A condition on an entity nobody declared is NOT dropped; the run
# aborts before any solving happens.
#
# The trap is the OFF-BY-ONE IN THE MESSAGE: the diagnostic reports the 0-based
# internal id and the half-open range, so `E: 5` against two declared surfaces
# reads "DSurface 4 not in range [0:2[".  Neither the number nor the bracket
# matches what was written in the deck.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = entity id used by the Dirichlet block, $2 = out file
cat > "$2" <<YAML
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
  - E: $1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 1, 0]
    FUNCT: [0, 0, 0]
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

mk 1 "$TMP/good.yaml"   # DSURFACE 1 exists in the topology
mk 5 "$TMP/bad.yaml"    # only DSURFACE 1 and 2 were declared

probe GOOD "$TMP/good.yaml"
probe BAD  "$TMP/bad.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
# The deck said E: 5; the diagnostic says DSurface 4, and the range is half-open.
grep -m1 -F "DSurface 4 not in range [0:2[" "$TMP/BAD.log"
grep -m1 -F "DSurface condition on non existent DSurface?Could not read set from entity type." "$TMP/BAD.log"
grep -m1 -F "4C_fem_condition.cpp" "$TMP/BAD.log"
# The id actually written in the deck is never echoed back.
echo "BAD_ECHOES_THE_WRITTEN_ID=$(grep -c 'DSurface 5' "$TMP/BAD.log")"
# It is not silently dropped: nothing was solved and no result test ran.
echo "BAD_RAN_A_RESULT_TEST=$(grep -c 'is CORRECT' "$TMP/BAD.log")"
exit 0
