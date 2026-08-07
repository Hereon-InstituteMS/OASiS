#!/bin/bash
# Tier-2 for fourc::input_format#25 — the TYPE of a DESIGN ... NEUMANN
# CONDITION is validated in two different places with two different
# vocabularies. The parser accepts
#   Dead | Live | PressureGrad | orthopressure | pseudo_orthopressure
# but the 3D SOLID element implements only Live, orthopressure and
# pseudo_orthopressure — 'Dead' parses cleanly and then aborts at the
# first element evaluation, which is the dangerous case because the
# input file looks valid.
# Verified by execution 2026-08-03, 4C 2026.2.0-dev.
set -u
# Resolve the 4C binary: explicit override first, then the paths this
# repo has been verified against (2026-08-03 verification host runs
# 4C 2026.2.0-dev at /home/alexander/4C/build/4C).
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# stdbuf is mandatory here: 4C writes the result-test verdict to raw
# std::cout and MPI_Abort discards a block-buffered stdout, which is
# itself pitfall input_format#18.
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }

mk() {  # $1 = TYPE value, $2 = file
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
    TYPE: "$1"
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
YAML
}

probe() {  # $1 = label, $2 = TYPE, $3 = needle
  mk "$2" "$TMP/$1.4C.yaml"
  run4c "$TMP/$1.4C.yaml" "$TMP/o_$1" > "$TMP/$1.log" 2>&1
  echo "$1: EXIT=$?"
  grep -m1 -o "$3" "$TMP/$1.log" || echo "$1: NEEDLE_MISSING"
}

# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology from both bad arms: they
# carry TYPE "Live", which the parser accepts AND the 3D SOLID element
# implements.  Both then run to completion, `dead: EXIT=1` becomes `dead: EXIT=0`
# (a forbidden token), and neither the element-evaluation abort nor the parser's
# possible-values list is printed — so the fixture must go red.
MUTATE="${T2_MUTATE:-0}"
DEAD_TYPE=Dead
BOGUS_TYPE=Follower
if [ "$MUTATE" = "1" ]; then DEAD_TYPE=Live; BOGUS_TYPE=Live; fi

probe live      Live           "Finalised step 1 / 1"
probe dead      "$DEAD_TYPE"   "Unknown type of SurfaceNeumann condition"
probe bogus     "$BOGUS_TYPE"  "possible values: Dead|Live|PressureGrad|orthopressure|pseudo_orthopressure"
exit 0
