#!/bin/bash
# Tier-2 for fourc::beams#10 — which cell types each beam element registers, and
# what 4C says when you ask for one it does not.
#
# Rule (correct): BEAM3R -> LINE2/LINE3/LINE4/LINE5, BEAM3K -> LINE2/LINE3/LINE4,
# BEAM3EB -> LINE2 only.
#
# Signal (wrong): the entry quoted 'Unknown beam element cell type' from
# beam_factory.cpp. Neither the string nor the file exists in 4C. The real
# message comes from the generic element-definition table:
#
#     Element 'BEAM3R' does not seem to know cell type 'line6'.
#     .../core/fem/src/general/element/4C_fem_general_element_definition.cpp
#
# — note the LOWERCASE cell type, so grepping the log for the "LINE6" you wrote
# finds nothing. LINE6 is a perfectly valid 4C cell type (LINE7 is not, and gets
# a different error from cell_type_traits); it is simply not registered for beams.
#
# Three self-contained arms on one six-node straight beam. Getting the node
# count right matters: a LINE6 line carrying only two node IDs fails much
# earlier, in the value parser, with a message about 'MAT' that says nothing
# about cell types at all.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = the STRUCTURE ELEMENTS block
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1
  TOLRES: 1e-06
  MAXITER: 15
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [0, 0, 1, 0, 0, 0]
    VAL: [0, 0, 10, 0, 0, 0]
    FUNCT: [0, 0, 1, 0, 0, 0]
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
  - "NODE 6 DNODE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 2.0 0.0 0.0"
  - "NODE 3 COORD 4.0 0.0 0.0"
  - "NODE 4 COORD 6.0 0.0 0.0"
  - "NODE 5 COORD 8.0 0.0 0.0"
  - "NODE 6 COORD 10.0 0.0 0.0"
STRUCTURE ELEMENTS:
$1
MATERIALS:
  - MAT: 1
    MAT_BeamReissnerElastHyper:
      YOUNG: 1e+07
      SHEARMOD: 5e+06
      DENS: 1.0
      CROSSAREA: 0.031415926535897934
      SHEARCORR: 1
      MOMINPOL: 1.5707963267948968e-04
      MOMIN2: 7.853981633974484e-05
      MOMIN3: 7.853981633974484e-05
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_TIME: "t"
YAML
}

T15="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
T18="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

# Registered: BEAM3R LINE5 (five nodes, fifteen triad values) plus a LINE2 stub.
deck "  - \"1 BEAM3R LINE5 1 2 3 4 5 MAT 1 TRIADS $T15\"
  - \"2 BEAM3R LINE2 5 6 MAT 1 TRIADS 0 0 0 0 0 0\"" > "$TMP/line5.yaml"
# Not registered for BEAM3R: LINE6, spelled with the right six nodes and eighteen triads.
deck "  - \"1 BEAM3R LINE6 1 2 3 4 5 6 MAT 1 TRIADS $T18\"" > "$TMP/line6.yaml"
# Not registered for BEAM3EB, which is LINE2-only.
deck "  - \"1 BEAM3EB LINE3 1 2 3 MAT 1\"" > "$TMP/eb_line3.yaml"

probe LINE5   "$TMP/line5.yaml"
probe LINE6   "$TMP/line6.yaml"
probe EBLINE3 "$TMP/eb_line3.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/LINE5.log"
grep -m1 -F "Element 'BEAM3R' does not seem to know cell type 'line6'." "$TMP/LINE6.log"
grep -m1 -F "Element 'BEAM3EB' does not seem to know cell type 'line3'." "$TMP/EBLINE3.log"
grep -m1 -F "4C_fem_general_element_definition.cpp" "$TMP/LINE6.log"
# The quoted diagnostic and its claimed source file do not exist.
echo "CLAIMED_UNKNOWN_BEAM_CELLTYPE_TEXT=$(grep -ci 'Unknown beam element cell type' "$TMP/LINE6.log")"
echo "CLAIMED_BEAM_FACTORY_FILE=$(grep -ci 'beam_factory' "$TMP/LINE6.log")"
# The message echoes the cell type in lower case, not as written in the deck.
echo "DIAGNOSTIC_ECHOES_UPPERCASE_LINE6=$(grep -c "cell type 'LINE6'" "$TMP/LINE6.log")"
exit 0
