#!/bin/bash
# Tier-2 for fourc::beams#0 — the entry said beams CANNOT use Exodus mesh files
# and "inline YAML is the only path". That is false, and this fixture runs the
# counterexample: a six-node BEAM3R LINE2 cantilever whose geometry comes
# entirely from an Exodus file, clamped and loaded through Exodus node sets,
# solved to completion.
#
#     STRUCTURE GEOMETRY:
#       FILE: "beamline.exo"
#       ELEMENT_BLOCKS:
#         - ID: 1
#           BEAM3R:
#             LINE2: { MAT: 1, TRIADS: [0, 0, 0, 0, 0, 0] }
#
# 4C's Exodus reader maps the BAR2/BAR3 cell shapes onto line2/line3, which is
# exactly what beams need, and reports "5 cells of type line2".
#
# What IS true is narrower and worth knowing: the reader takes coordinates, cell
# blocks and node sets from the file and nothing else — it never reads nodal or
# cell variables. So the OTHER way of supplying nodal triads,
# NODAL_ROTATION_VECTORS: <field name>, cannot be satisfied from an Exodus file:
#
#     The cell data does not contain the key 'TRIADS'.
#     .../core/io/src/4C_io_mesh.hpp
#
# and dropping the triad source altogether fails the element spec outright. Use
# a literal TRIADS inside the element block; note that it is ONE triad set shared
# by every element of the block, which is fine for a straight beam and wrong for
# a curved one.
#
# beamline.exo ships with this fixture: six nodes on the x-axis at 0,2,...,10,
# five BAR2 cells, node sets 1 ("clamped", node 1) and 2 ("tip", node 6). It was
# written with meshio and then patched with netCDF4 to give ns_prop1/eb_prop1
# the name="ID" attribute and one-based IDs, which meshio omits and the Exodus
# library requires.
. "$(dirname "$0")/../_lib/preamble.sh"

MESH="$(dirname "$0")/beamline.exo"
[ -f "$MESH" ] || { echo "FIXTURE_ABORT=missing_exodus_asset"; exit 3; }
cp "$MESH" "$TMP/beamline.exo"

deck() {  # $1 = the triad-source line inside the LINE2 block
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1
  TOLRES: 1e-06
  MAXITER: 25
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
STRUCTURE GEOMETRY:
  FILE: "beamline.exo"
  SHOW_INFO: "detailed_summary"
  ELEMENT_BLOCKS:
    - ID: 1
      BEAM3R:
        LINE2:
          MAT: 1
$1
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 6
    ONOFF: [0, 0, 1, 0, 0, 0]
    VAL: [0, 0, 2, 0, 0, 0]
    FUNCT: [0, 0, 1, 0, 0, 0]
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

deck "          TRIADS: [0, 0, 0, 0, 0, 0]"     > "$TMP/literal.yaml"
deck "          NODAL_ROTATION_VECTORS: TRIADS" > "$TMP/from_mesh.yaml"
deck ""                                        > "$TMP/no_triads.yaml"

# 4C resolves STRUCTURE GEOMETRY: FILE: relative to the working directory.
cd "$TMP" || exit 3
probe LITERAL  "$TMP/literal.yaml"
probe FROMMESH "$TMP/from_mesh.yaml"
probe NOTRIADS "$TMP/no_triads.yaml"

# The counterexample: beams read from an Exodus file, solved to the end.
grep -m1 -F "Read mesh from file" "$TMP/LITERAL.log"
grep -m1 -F "Mesh consists of 6 points and 5 cells organized in 1 cell-blocks and 2 point-sets." "$TMP/LITERAL.log"
grep -m1 -F "cells of type line2" "$TMP/LITERAL.log"
grep -m1 -F "processor 0 finished normally" "$TMP/LITERAL.log"
echo "EXODUS_BEAM_STEPS=$(grep -c 'Finalised step' "$TMP/LITERAL.log")"

# The real limitation: no cell/point data comes out of the Exodus reader.
grep -m1 -F "The cell data does not contain the key 'TRIADS'." "$TMP/FROMMESH.log"
grep -m1 -F "4C_io_mesh.hpp" "$TMP/FROMMESH.log"
# ...and a block with no triad source at all does not match the element spec.
grep -m1 -F "Could not match this input" "$TMP/NOTRIADS.log"

# None of the three logs contains the diagnostic the entry quoted.
python3 - "$TMP/LITERAL.log" "$TMP/FROMMESH.log" "$TMP/NOTRIADS.log" <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += t.count("not supported in exodus") + t.count("4c_io_meshreader.cpp")
print("CLAIMED_EXODUS_REFUSAL_TEXT=%d" % n)
PY
exit 0
