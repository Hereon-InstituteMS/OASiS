#!/bin/bash
# Tier-2 for fourc::ale#0 — stand-alone ALE meshes are declared in
# `ALE ELEMENTS` with the ALE2 (2D) / ALE3 (3D) element category, and
# `SOLID` is not a substitute. Two arms on one self-contained 2D deck:
#
#   good : "1 ALE2  QUAD4 1 2 3 4 MAT 1"  -> runs to completion
#   bad  : "1 SOLID QUAD4 1 2 3 4 MAT 1"  -> aborts at element definition
#
# The interesting part is WHERE it fails: the diagnostic names the CELL
# type, not the element category, so a reader who greps for "ALE" in the
# error learns nothing. The exact text is asserted below.
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -n "${_c:-}" ] && [ -x "$_c" ] && BIN="$_c" && break
done
if [ -z "${BIN:-}" ]; then echo "FIXTURE_ABORT=no_4c_binary"; exit 3; fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# stdbuf: 4C writes its verdicts to raw std::cout and MPI_Abort discards a
# block-buffered stdout (pitfall input_format#18).
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }

deck() {  # $1 = the two element lines' category token
cat <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Ale"
ALE DYNAMIC:
  TIMESTEP: 0.25
  NUMSTEP: 2
  MAXTIME: 0.5
  MAXITER: 10
  TOLRES: 1e-08
  TOLDISP: 1e-08
  RESULTSEVERY: 1
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [5]
      DENS: 500
  - MAT: 5
    ELAST_CoupNeoHooke:
      YOUNG: 250
      NUE: 0.3
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
  - E: 2
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 1]
    FUNCT: [0, 1]
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
  - "NODE 2 DNODE 1"
  - "NODE 5 DNODE 2"
  - "NODE 6 DNODE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 2.0 0.0"
  - "NODE 6 COORD 1.0 2.0 0.0"
ALE ELEMENTS:
  - "1 $1 QUAD4 1 2 3 4 MAT 1"
  - "2 $1 QUAD4 4 3 6 5 MAT 1"
YAML
}

deck "ALE2"  > "$TMP/good.yaml"
deck "SOLID" > "$TMP/bad.yaml"

run4c "$TMP/good.yaml" "$TMP/o_good" > "$TMP/good.log" 2>&1
echo "EXIT_ALE2=$?"
run4c "$TMP/bad.yaml" "$TMP/o_bad" > "$TMP/bad.log" 2>&1
echo "EXIT_SOLID=$?"

grep -m1 -F "processor 0 finished normally" "$TMP/good.log"
grep -m1 -F "Element 'SOLID' does not seem to know cell type 'quad4'." "$TMP/bad.log"
grep -m1 -F "4C_fem_general_element_definition.cpp" "$TMP/bad.log"
# The diagnostic must NOT mention ALE at all — that is the trap.
if grep -qiE "expected ALE element|ale_factory|ALE element type" "$TMP/bad.log"; then
  echo "DIAGNOSTIC_MENTIONS_ALE=yes"
else
  echo "DIAGNOSTIC_MENTIONS_ALE=no"
fi
exit 0
