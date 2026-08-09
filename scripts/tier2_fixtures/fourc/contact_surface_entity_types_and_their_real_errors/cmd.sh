#!/bin/bash
# Tier-2 for fourc::contact#6 — how a mortar contact surface is identified, and
# what 4C says for each wrong way of doing it.
#
# Claimed:  parser error 'expected node set for contact surface, got element set'
#           / 'MortarInterface needs DNODE not DELE'.
# Observed: neither string is in 4C, and no message of that shape is printed.
#
# What is actually there: a condition carries an optional ENTITY_TYPE, one of
# legacy_id (the default, an index into the DSURF/DLINE/DNODE-NODE TOPOLOGY
# lists), node_set_id, element_block_id, or a NODE_SET_NAME. The last three all
# resolve against a MESH FILE, so on a deck with inline topology they fail with
# the same shape of message naming the entity kind:
#
#     Cannot apply condition 'Contact' to element block 2 which is not specified in the mesh file.
#     Cannot apply condition 'Contact' to node set 2 which is not specified in the mesh file.
#     .../core/fem/src/condition/4C_fem_condition.cpp
#
# and supplying the topology as volumes instead of surfaces fails earlier, with
# the design-entity range message:
#
#     DSurface 0 not in range [0:0[
#
# The entry's advice is right in substance — a contact surface is a set of NODES
# forming a design surface — but element_block_id is a real ENTITY_TYPE, not a
# parse error, and it exists precisely to point a condition at an element block.
# The refusal below is about the mesh file being absent, not about element sets
# being forbidden.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra line inside each contact condition, $2 = topology keyword pair
cat <<YAML
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
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
$1    InterfaceID: 1
    Side: "Master"
  - E: 3
$1    InterfaceID: 1
    Side: "Slave"
$2-NODE TOPOLOGY:
  - "NODE 1 $3 1"
  - "NODE 2 $3 1"
  - "NODE 3 $3 1"
  - "NODE 4 $3 1"
  - "NODE 5 $3 2"
  - "NODE 6 $3 2"
  - "NODE 7 $3 2"
  - "NODE 8 $3 2"
  - "NODE 9 $3 3"
  - "NODE 10 $3 3"
  - "NODE 11 $3 3"
  - "NODE 12 $3 3"
  - "NODE 13 $3 4"
  - "NODE 14 $3 4"
  - "NODE 15 $3 4"
  - "NODE 16 $3 4"
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

deck ""                                  DSURF DSURFACE > "$TMP/legacy.yaml"
deck "    ENTITY_TYPE: element_block_id
" DSURF DSURFACE > "$TMP/element_block.yaml"
deck "    ENTITY_TYPE: node_set_id
" DSURF DSURFACE > "$TMP/node_set.yaml"
deck ""                                  DVOL  DVOLUME  > "$TMP/volume_topology.yaml"

probe LEGACY      "$TMP/legacy.yaml"
probe ELEMENTBLOCK "$TMP/element_block.yaml"
probe NODESET     "$TMP/node_set.yaml"
probe VOLTOPOLOGY "$TMP/volume_topology.yaml"

# The default, inline way works.
grep -m1 -F "Building contact interface" "$TMP/LEGACY.log"
grep -m1 -F "processor 0 finished normally" "$TMP/LEGACY.log"

# element_block_id is a real ENTITY_TYPE; it just needs a mesh file.
grep -m1 -F "Cannot apply condition 'Contact' to element block 2 which is not specified in the mesh file." "$TMP/ELEMENTBLOCK.log"
grep -m1 -F "Cannot apply condition 'Contact' to node set 2 which is not specified in the mesh file." "$TMP/NODESET.log"
grep -m1 -F "4C_fem_condition.cpp" "$TMP/ELEMENTBLOCK.log"
# Declaring the same node groups as volumes leaves the design surfaces empty.
grep -m1 -F "not in range [0:0[" "$TMP/VOLTOPOLOGY.log"
grep -m1 -F "DSurface condition on non existent DSurface?Could not read set from entity type." "$TMP/VOLTOPOLOGY.log"

python3 - "$TMP/ELEMENTBLOCK.log" "$TMP/NODESET.log" "$TMP/VOLTOPOLOGY.log" <<'PY'
import sys
n = 0
for p in sys.argv[1:]:
    t = open(p, "rb").read().decode("utf-8", "replace").lower()
    n += (t.count("expected node set for contact surface")
          + t.count("mortarinterface needs dnode")
          + t.count("got element set"))
print("CLAIMED_ENTITY_TYPE_TEXTS=%d" % n)
PY
exit 0
