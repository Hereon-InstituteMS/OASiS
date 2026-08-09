#!/bin/bash
# Tier-2 for fourc::structural_dynamics#9 — RESTARTEVERY exists in BOTH
# the IO section and STRUCTURAL DYNAMIC, but only the STRUCTURAL DYNAMIC
# one is read by the structural integrator. Placed under IO the run
# still exits 0 and writes no restart record; the mistake only surfaces
# when --restart is attempted.
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

mk() {  # $1 = 'io' or 'sdyn', $2 = file
local IOBLOCK="" SDYN=""
if [ "$1" = "io" ]; then IOBLOCK=$'IO:\n  RESTARTEVERY: 2'; else SDYN=$'\n  RESTARTEVERY: 2'; fi
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 4
  MAXTIME: 0.4
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1$SDYN
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
$IOBLOCK
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

probe() {  # $1 = label, $2 = placement
  local d="$TMP/$1"; mkdir -p "$d"
  mk "$2" "$d/in.4C.yaml"
  run4c "$d/in.4C.yaml" "$d/out" > "$d/log1" 2>&1
  local rc1=$?
  local n; n=$(find "$d" -name 'out.result.structure.s*' | wc -l)
  echo "$1: WRITE_EXIT=$rc1 RESTART_RECORDS=$n"
  stdbuf -oL -eL "$BIN" --restart=2 "$d/in.4C.yaml" "$d/out" > "$d/log2" 2>&1
  echo "$1: RESTART_EXIT=$?"
  grep -m1 -o "No restart entry for discretization 'structure' step 2" "$d/log2" || true
  grep -m1 -o "Restart of the structural simulation from step 2" "$d/log2" || true
}

# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology: the arm LABELLED
# io_section is built with RESTARTEVERY in STRUCTURAL DYNAMIC, the placement that
# works.  It then writes a restart record and restarts cleanly, so
# `io_section: RESTART_EXIT=1` and "No restart entry for discretization
# 'structure' step 2" disappear and the forbidden
# `io_section: WRITE_EXIT=0 RESTART_RECORDS=1` appears — the fixture must go red.
MUTATE="${T2_MUTATE:-0}"
IO_ARM_PLACEMENT=io
[ "$MUTATE" = "1" ] && IO_ARM_PLACEMENT=sdyn

probe io_section   "$IO_ARM_PLACEMENT"
probe sdyn_section sdyn
exit 0
