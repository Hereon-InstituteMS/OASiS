#!/bin/bash
# Tier-2 for fourc::input_format#26 — CORRECTION fixture.
#
# The sibling fixture runtime_vtk_output_needs_both_sections proves that
# BOTH IO/RUNTIME VTK OUTPUT (INTERVAL_STEPS > 0) and the per-field
# sub-section IO/RUNTIME VTK OUTPUT/STRUCTURE are needed.  It does so
# with a sub-section that already carries DISPLACEMENT: true — but the
# pitfall PROSE only ever said "IO/RUNTIME VTK OUTPUT/STRUCTURE with
# OUTPUT_STRUCTURE: true".  An agent following the prose writes the
# sub-section without any field flag and gets a crash, not 3 .vtu files.
#
# Claim under test (verified by execution 2026-08-03, 4C 2026.2.0-dev
# git 89519cf), 2-step single-HEX8 deck:
#
#   OUTPUT_STRUCTURE + DISPLACEMENT -> EXIT=0 VTU=3   (the working form)
#   OUTPUT_STRUCTURE only           -> EXIT=1         "No data was
#                                      written or writer was already in
#                                      final phase." from
#                                      4C_io_vtk_writer_base
#   DISPLACEMENT only               -> EXIT=0 VTU=0   (silent no-op:
#                                      OUTPUT_STRUCTURE is the master
#                                      switch)
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

BODY='PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 2
  MAXTIME: 0.2
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
      DENS: 1'

run_case() {  # $1 = label, $2 = the IO preamble
  local d="$TMP/$1"; mkdir -p "$d"
  printf '%s\n%s\n' "$2" "$BODY" > "$d/in.4C.yaml"
  stdbuf -oL -eL "$BIN" "$d/in.4C.yaml" "$d/out" > "$d/log.txt" 2>&1
  local rc=$?
  local n; n=$(find "$d" -name '*.vtu' | wc -l)
  local throw=no
  grep -q "No data was written or writer was already in final phase" "$d/log.txt" && throw=yes
  echo "$1: EXIT=$rc VTU=$n NO_DATA_THROW=$throw"
}

run_case with_displacement 'IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true'

run_case output_structure_only 'IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true'

run_case displacement_only 'IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  DISPLACEMENT: true'

exit 0
