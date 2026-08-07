#!/bin/bash
# Tier-2 for fourc::structural_mechanics / fourc::thermal — the STRUCTURE
# ELEMENTS / THERMO ELEMENTS line grammar.
#
# Pins three things a reader cannot get from `4C --parameters` alone:
#  1. the REQUIRED-key sets really are enforced (SOLID: MAT+KINEM;
#     THERMO: MAT and NOTHING else);
#  2. being listed by `--parameters` is NOT the same as working — THERMO
#     TRI6 parses and then dies at element evaluation;
#  3. INTEGRATION cannot be written on a legacy element line in ANY syntax,
#     and a PARTIAL INTEGRATION produces no 4C diagnostic at all: it
#     core-dumps with std::bad_any_cast and shell status 134.
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }

struct() {  # $1 = element line
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
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
  - "$1"
YAML
}

thermo2d() {  # $1 = element line
cat <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: "OneStepTheta"
  TIMESTEP: 0.1
  NUMSTEP: 2
  MAXTIME: 0.2
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/ONESTEPTHETA:
  THETA: 1.0
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "T"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [1.0]
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [100.0]
    FUNCT: [0]
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 2 DLINE 1"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.5 0.0 0.0"
  - "NODE 6 COORD 1.0 0.5 0.0"
  - "NODE 7 COORD 0.5 0.5 0.0"
THERMO ELEMENTS:
  - "$1"
YAML
}

probe() {  # $1 label  $2 builder  $3 element line
  "$2" "$3" > "$TMP/$1.4C.yaml"
  run4c "$TMP/$1.4C.yaml" "$TMP/o_$1" > "$TMP/$1.log" 2>&1
  echo "EXIT_$1=$?"
}

# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology from every bad arm: each
# is given the SAME well-formed element line as its own OK arm — "1 SOLID HEX8
# ... MAT 1 KINEM nonlinear" for the structural probes, "1 THERMO QUAD4 1 2 3 4
# MAT 1" for the thermal ones.  Every EXIT_*=1 expectation becomes 0, EXIT_S_INT_
# PART=134 becomes 0, and both forbidden tokens EXIT_S_HEX8_OK=1 / EXIT_T_QUAD4_
# OK=1 stay absent while every quoted diagnostic disappears — the fixture must go
# red.  S_INT_PART_FOURC_ERROR_BLOCKS=0 is the one expectation that survives: it
# counts an ABSENCE, and a clean run has no error block either.  That is exactly
# why it is a weak assertion on its own and why the other ten carry the fixture.
MUTATE="${T2_MUTATE:-0}"
S_GOOD="1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
T_GOOD="1 THERMO QUAD4 1 2 3 4 MAT 1"
S_NOKINEM_L="1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1"
S_QUAD4_L="1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM nonlinear"
S_BOGUS_L="1 FROBNICATE HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
S_TOTLAG_L="1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinearTotLag"
S_TECH_L="1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear TECH banana"
S_INT_BOTH_L="1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear INTEGRATION RESIDUUM hex_27point MASS hex_8point"
S_INT_PART_L="1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear INTEGRATION RESIDUUM hex_27point"
T_EXTRAKEY_L="1 THERMO QUAD4 1 2 3 4 MAT 1 KINEM linear"
T_TRI6_L="1 THERMO TRI6 1 2 3 5 6 7 MAT 1"
if [ "$MUTATE" = "1" ]; then
  S_NOKINEM_L="$S_GOOD"; S_QUAD4_L="$S_GOOD"; S_BOGUS_L="$S_GOOD"
  S_TOTLAG_L="$S_GOOD"; S_TECH_L="$S_GOOD"; S_INT_BOTH_L="$S_GOOD"
  S_INT_PART_L="$S_GOOD"; T_EXTRAKEY_L="$T_GOOD"; T_TRI6_L="$T_GOOD"
fi

probe S_HEX8_OK    struct   "$S_GOOD"
probe S_NOKINEM    struct   "$S_NOKINEM_L"
probe S_QUAD4      struct   "$S_QUAD4_L"
probe S_BOGUS      struct   "$S_BOGUS_L"
probe S_TOTLAG     struct   "$S_TOTLAG_L"
probe S_TECH_TET   struct   "$S_TECH_L"
probe S_INT_BOTH   struct   "$S_INT_BOTH_L"
probe S_INT_PART   struct   "$S_INT_PART_L"
probe T_QUAD4_OK   thermo2d "$T_GOOD"
probe T_EXTRAKEY   thermo2d "$T_EXTRAKEY_L"
probe T_TRI6_DEAD  thermo2d "$T_TRI6_L"

# The partial-INTEGRATION arm must produce NO 4C error block at all.
echo "S_INT_PART_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/S_INT_PART.log")"

for a in S_NOKINEM S_QUAD4 S_BOGUS S_TOTLAG S_TECH_TET S_INT_BOTH S_INT_PART T_EXTRAKEY T_TRI6_DEAD; do
  echo "--- $a ---"
  grep -m1 -h -F -e "Required value 'KINEM' not found in input line" \
      -e "Element 'SOLID' does not seem to know cell type 'quad4'." \
      -e "Unknown type 'FROBNICATE' of finite element" \
      -e "Could not parse parameter 'KINEM': invalid value 'nonlinearTotLag'" \
      -e "Could not parse value 'banana' as an enum constant" \
      -e "Key 'INTEGRATION' cannot be found in the container." \
      -e "terminate called after throwing an instance of 'std::bad_any_cast'" \
      -e "After parsing, the line still contains 'KINEM linear'." \
      -e "Element shape TRI6 (6 nodes) not activated. Just do it." "$TMP/$a.log"
done
exit 0
