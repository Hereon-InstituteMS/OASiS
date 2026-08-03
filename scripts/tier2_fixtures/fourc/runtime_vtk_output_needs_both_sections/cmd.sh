#!/bin/bash
# Tier-2 for fourc::input_format#26 — runtime VTK output is silently
# skipped unless BOTH the parent IO/RUNTIME VTK OUTPUT section (with a
# positive INTERVAL_STEPS, which defaults to -1 = never) AND the
# per-field IO/RUNTIME VTK OUTPUT/STRUCTURE sub-section are present.
# Every variant below exits 0; only the complete one writes files.
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

mk() {  # $1 = IO block, $2 = file
cat > "$2" <<YAML
PROBLEM TYPE:
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
$1
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 1"
  - "NODE 8 DSURFACE 1"
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

probe() {  # $1 = label, $2 = IO block
  local d="$TMP/$1"; mkdir -p "$d"
  mk "$2" "$d/in.4C.yaml"
  run4c "$d/in.4C.yaml" "$d/out" > "$d/log" 2>&1
  local rc=$?
  local n; n=$(find "$d" -name '*.vtu' | wc -l)
  echo "$1: EXIT=$rc VTU=$n"
}

FULL='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true'
PARENT_ONLY='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1'
NO_INTERVAL='IO/RUNTIME VTK OUTPUT:
  OUTPUT_DATA_FORMAT: ascii
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true'

probe both_sections "$FULL"
probe parent_only "$PARENT_ONLY"
probe no_interval_steps "$NO_INTERVAL"
probe no_io_at_all "IO:
  VERBOSITY: standard"
exit 0
