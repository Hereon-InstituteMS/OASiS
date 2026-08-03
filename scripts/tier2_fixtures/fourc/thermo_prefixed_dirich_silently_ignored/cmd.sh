#!/bin/bash
# Tier-2 for fourc::thermal#3 — in a standalone PROBLEMTYPE: Thermo run
# the section 'DESIGN SURF THERMO DIRICH CONDITIONS' parses cleanly and
# is then silently dropped; the temperature stays at 0. The section that
# actually reaches the thermo discretisation is the PLAIN
# 'DESIGN SURF DIRICH CONDITIONS'. The two runs below differ in that
# one word and in nothing else.
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

mk() {  # $1 = dirich section name, $2 = file
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "T"
$1:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [100.0]
    FUNCT: [0]
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
THERMO ELEMENTS:
  - "1 THERMO HEX8 1 2 3 4 5 6 7 8 MAT 1"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [1.0]
RESULT DESCRIPTION:
  - THERMAL:
      DIS: "thermo"
      NODE: 2
      QUANTITY: "temp"
      VALUE: 100.0
      TOLERANCE: 1e-8
YAML
}

echo "=== THERMO-prefixed section (silently ignored) ==="
mk "DESIGN SURF THERMO DIRICH CONDITIONS" "$TMP/pref.4C.yaml"
run4c "$TMP/pref.4C.yaml" "$TMP/o_pref" | grep -E "is CORRECT|is WRONG|not a valid section" | head -2
echo "EXIT_PREFIXED=${PIPESTATUS[0]}"

echo "=== plain section (works) ==="
mk "DESIGN SURF DIRICH CONDITIONS" "$TMP/plain.4C.yaml"
run4c "$TMP/plain.4C.yaml" "$TMP/o_plain" | grep -E "is CORRECT|is WRONG" | head -2
echo "EXIT_PLAIN=${PIPESTATUS[0]}"
exit 0
