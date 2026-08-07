#!/bin/bash
# Tier-2 for fourc::input_format#22 — a FUNCT index used by a condition must
# resolve to a function that 4C actually READ, and 4C does not fall back to a
# constant.  Two things about how the index is resolved are not obvious:
#
#   * The index is POSITIONAL, not the number in the section name.  4C reads
#     FUNCT1, FUNCT2, ... consecutively and stops at the first gap, so a deck
#     that defines FUNCT1 and FUNCT7 and points a condition at FUNCT 7 fails
#     with the SAME message as a deck that defines no function at all: the lone
#     FUNCT7 was never read.
#   * The failure is raised at the FIRST EVALUATION, not at parse time — the
#     input file is read and the discretisation is built first.
#
# FUNCT 0 is not an index; it means "no function, use VAL directly".
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = FUNCT sections (may be empty), $2 = index in the Neumann block, $3 = out
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
$1
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
    VAL: [0, 1, 0]
    FUNCT: [0, $2, 0]
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

ONE_TO_SEVEN='FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
FUNCT3:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
FUNCT4:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
FUNCT5:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
FUNCT6:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
FUNCT7:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"'
ONE_AND_SEVEN='FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
FUNCT7:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"'

mk ''                 0 "$TMP/zero.yaml"     # FUNCT 0: no function, VAL used directly
mk "$ONE_TO_SEVEN"    7 "$TMP/all7.yaml"     # FUNCT1..7 defined, index 7 resolves
mk "$ONE_AND_SEVEN"   7 "$TMP/gap.yaml"      # FUNCT7 defined but unreachable behind a gap
mk ''                 7 "$TMP/none.yaml"     # index 7 with no FUNCT sections at all

probe ZERO "$TMP/zero.yaml"
probe ALL7 "$TMP/all7.yaml"
probe GAP  "$TMP/gap.yaml"
probe NONE "$TMP/none.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/ZERO.log"
# A resolvable FUNCT7 returning 1.0 reproduces the FUNCT-0 answer exactly.
grep -m1 -F "is CORRECT, abs(diff)= 0.00000000000000000e+00" "$TMP/ALL7.log"
grep -m1 -F "Function with index 7 (i.e. input FUNCT7) not available." "$TMP/GAP.log"
grep -m1 -F "4C_utils_function_manager.hpp" "$TMP/GAP.log"
# Defining FUNCT7 behind a gap is indistinguishable from not defining it at all.
echo "GAP_AND_NONE_SAY_THE_SAME=$([ "$(grep -c 'Function with index 7 (i.e. input FUNCT7) not available.' "$TMP/GAP.log")" = "$(grep -c 'Function with index 7 (i.e. input FUNCT7) not available.' "$TMP/NONE.log")" ] && echo yes || echo no)"
# Raised after the input was read and the mesh built, not by the parser.
echo "GAP_REACHED_FILL_COMPLETE=$(grep -c 'fill_complete() on discretization structure' "$TMP/GAP.log")"
echo "GAP_BLAMED_THE_PARSER=$(grep -c '4C_io_input_spec_builders.cpp' "$TMP/GAP.log")"
exit 0
