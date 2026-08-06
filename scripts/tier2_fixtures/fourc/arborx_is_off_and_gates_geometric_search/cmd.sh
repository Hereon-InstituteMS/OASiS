#!/bin/bash
# Tier-2 for fourc::input_format#15 — ArborX is an optional dependency, it is
# OFF in this build, and what it gates is the GEOMETRIC SEARCH, not the linear
# solver.
#
# The observable is the abort you get when you ask for the unavailable thing.
# MESH PARTITIONING / METHOD: monolithic is the cheapest input-level route into
# Core::GeometricSearch: it builds a monolithic node graph via a global
# collision search, which needs a BoundingVolume, which is #ifdef'd out without
# ArborX.  The other two partitioning methods do not touch geometric search and
# run to the same answer, which is the point: turning ArborX off costs you a
# search path and nothing else.
#
# The build flag is read from the CMakeCache next to the binary when it is
# there, but the fixture does not depend on it -- the abort is proof enough.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = MESH PARTITIONING method, $2 = out file
cat > "$2" <<YAML
MESH PARTITIONING:
  METHOD: $1
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

mk hypergraph  "$TMP/hypergraph.yaml"
mk multijagged "$TMP/multijagged.yaml"
mk monolithic  "$TMP/monolithic.yaml"

probe HYPERGRAPH  "$TMP/hypergraph.yaml"
probe MULTIJAGGED "$TMP/multijagged.yaml"
probe MONOLITHIC  "$TMP/monolithic.yaml"

# The two search-free partitioners run and agree on the answer.
grep -m1 -F "Redistributing using hypergraph" "$TMP/HYPERGRAPH.log"
grep -m1 -F "Redistributing using recursive coordinate bisection" "$TMP/MULTIJAGGED.log"
grep -m1 -F "processor 0 finished normally" "$TMP/HYPERGRAPH.log"
echo "HYPERGRAPH_CORRECT=$(grep -c 'is CORRECT' "$TMP/HYPERGRAPH.log")"
echo "MULTIJAGGED_CORRECT=$(grep -c 'is CORRECT' "$TMP/MULTIJAGGED.log")"

# The one that needs geometric search gets as far as announcing itself, then
# dies in the bounding-volume constructor.
grep -m1 -F "Redistributing using monolithic hypergraph" "$TMP/MONOLITHIC.log"
grep -m1 -F "The struct 'Core::GeometricSearch::BoundingVolume' can only be used with ArborX.To use it, enable ArborX during the configure process." "$TMP/MONOLITHIC.log"
grep -m1 -F "4C_geometric_search_bounding_volume.hpp" "$TMP/MONOLITHIC.log"
# It is the SEARCH that is missing, not the solver: the failing arm never got
# to a linear solve at all.
echo "MONOLITHIC_REACHED_A_LINEAR_SOLVE=$(grep -c 'Core::LinAlg::Solver' "$TMP/MONOLITHIC.log")"
echo "HYPERGRAPH_REACHED_A_LINEAR_SOLVE=$(grep -c 'Core::LinAlg::Solver' "$TMP/HYPERGRAPH.log")"

# The build flag itself, when the cache is next to the binary.
CACHE="$(dirname "$BIN")/CMakeCache.txt"
if [ -f "$CACHE" ]; then
  echo "CMAKECACHE_ARBORX=$(grep -m1 '^FOUR_C_WITH_ARBORX:BOOL=' "$CACHE")"
else
  echo "CMAKECACHE_ARBORX=not_present_next_to_binary"
fi
exit 0
