#!/bin/bash
# Tier-2 for fourc::structural_mechanics#9 and #10 —
#   #9 DENS is a REQUIRED parameter of MAT_Struct_StVenantKirchhoff in
#      every analysis type, including DYNAMICTYPE: Statics;
#   #10 NUE is validated against the half-open range [-1, 0.5), so the
#      incompressible limit NUE: 0.5 is rejected outright.
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

mk() {  # $1 = material body, $2 = file
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
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
$1
YAML
}

probe() {  # $1 = label, $2 = material body, $3 = needle
  mk "$2" "$TMP/$1.4C.yaml"
  # Capture to a file first: piping straight into `grep -m1` makes grep
  # close the pipe early and the exit status becomes SIGPIPE (141),
  # which would misreport a perfectly healthy 4C run.
  run4c "$TMP/$1.4C.yaml" "$TMP/o_$1" > "$TMP/$1.log" 2>&1
  echo "$1: EXIT=$?"
  grep -m1 -o "$3" "$TMP/$1.log" || echo "$1: NEEDLE_MISSING"
}

echo "=== NUE 0.49 (inside range) — must run ==="
probe nue_ok "      YOUNG: 1000
      NUE: 0.49
      DENS: 1" "fill_complete() on discretization structure"

echo "=== NUE 0.5 (incompressible limit) — must be rejected ==="
probe nue_half "      YOUNG: 1000
      NUE: 0.5
      DENS: 1" "Candidate parameter 'NUE' does not pass validation: in_range\\[-1,0.5)"

echo "=== DENS omitted under Statics — must be rejected ==="
probe dens_missing "      YOUNG: 1000
      NUE: 0.3" "Expected parameter 'DENS'"
exit 0
