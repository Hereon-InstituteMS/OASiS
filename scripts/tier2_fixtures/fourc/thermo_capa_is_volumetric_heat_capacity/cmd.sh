#!/bin/bash
# Tier-2 for fourc::thermo#3 — MAT_Fourier.CAPA is the VOLUMETRIC heat
# capacity rho*c_p, not the specific heat c_p, and getting it wrong is a
# silent factor-of-rho error in the thermal time constant.
#
# The probe is a 2x2 QUAD4 unit square with the whole boundary held at 0 and
# an initial field sin(pi x) sin(pi y).  Exactly ONE interior degree of
# freedom survives the Dirichlet conditions, so the discrete problem is the
# scalar ODE  CAPA * m * du/dt = -k * u  and the decay constant is strictly
# proportional to CAPA — nothing else in the deck can absorb the error.
#
#   VOLUMETRIC    CAPA = rho*c_p = 7850  , t_end = 392.5   -> centre 0.3012
#   SPECIFIC_ONLY CAPA = c_p     = 1     , t_end = 392.5   -> centre 4.2e-08
#   RESCALED      CAPA = c_p     = 1     , t_end = 392.5/7850 -> centre 0.3012
#
# The third arm is the quantitative statement: the SAME centre temperature is
# reached after a time shorter by exactly the factor rho.  All three arms
# assert the same RESULT DESCRIPTION value, so 4C itself is the judge.
# --- self-contained preamble (deliberately NOT sourced from ../_lib) --------
# scripts/mutate_tier2_fixtures.py copies ONLY this directory into a scratch
# tree.  A fixture that sources ../_lib/preamble.sh therefore cannot even
# start there, its mutant dies for the wrong reason, and the KILLED verdict
# certifies nothing.  Everything this fixture needs is inline, so the
# mutation proof is real.  Same honesty rule as the shared preamble: when 4C
# is missing this prints FIXTURE_ABORT=no_binary and exits non-zero, and
# fixture.json forbids both strings, so an absent solver makes the fixture
# RED rather than green.
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" \
          "$HOME/Schreibtisch/4C-src/4C/build/4C" "/usr/local/bin/4C"; do
  [ -n "${_c:-}" ] && [ -x "$_c" ] && BIN="$_c" && break
done
if [ -z "${BIN:-}" ]; then
  echo "FIXTURE_ABORT=no_binary (set FOURC_BINARY to a 4C executable)"
  exit 3
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# stdbuf is not decoration: 4C writes result-test verdicts to raw std::cout
# and MPI_Abort discards a block-buffered stdout (pitfall input_format#18).
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }
probe() { run4c "$2" "$TMP/o_$1" > "$TMP/$1.log" 2>&1; echo "EXIT_$1=$?"; }
# ---------------------------------------------------------------------------

deck() {  # $1 = CAPA, $2 = TIMESTEP, $3 = NUMSTEP, $4 = MAXTIME
cat <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: OneStepTheta
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: $2
  NUMSTEP: $3
  MAXTIME: $4
  RESULTSEVERY: $3
  RESTARTEVERY: 0
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/ONESTEPTHETA:
  THETA: 0.5
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "T"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: $1
      CONDUCT:
        constant: [1.0]
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "sin(pi*x)*sin(pi*y)"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
RESULT DESCRIPTION:
  - THERMAL:
      DIS: "thermo"
      NODE: 5
      QUANTITY: "temp"
      VALUE: 0.301193127609127531
      TOLERANCE: 1e-6
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 2 DLINE 1"
  - "NODE 3 DLINE 1"
  - "NODE 4 DLINE 1"
  - "NODE 6 DLINE 1"
  - "NODE 7 DLINE 1"
  - "NODE 8 DLINE 1"
  - "NODE 9 DLINE 1"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 0.5 0.0 0.0"
  - "NODE 3 COORD 1.0 0.0 0.0"
  - "NODE 4 COORD 0.0 0.5 0.0"
  - "NODE 5 COORD 0.5 0.5 0.0"
  - "NODE 6 COORD 1.0 0.5 0.0"
  - "NODE 7 COORD 0.0 1.0 0.0"
  - "NODE 8 COORD 0.5 1.0 0.0"
  - "NODE 9 COORD 1.0 1.0 0.0"
THERMO ELEMENTS:
  - "1 THERMO QUAD4 1 2 5 4 MAT 1"
  - "2 THERMO QUAD4 2 3 6 5 MAT 1"
  - "3 THERMO QUAD4 4 5 8 7 MAT 1"
  - "4 THERMO QUAD4 5 6 9 8 MAT 1"
YAML
}

# rho = 7850, c_p = 1  =>  CAPA must be 7850, and the transient takes 392.5 s.
deck 7850.0 1.9625     200 392.5 > "$TMP/volumetric.yaml"
deck 1.0    1.9625     200 392.5 > "$TMP/specific_only.yaml"
deck 1.0    0.00025    200 0.05  > "$TMP/rescaled.yaml"

probe VOLUMETRIC    "$TMP/volumetric.yaml"
probe SPECIFIC_ONLY "$TMP/specific_only.yaml"
probe RESCALED      "$TMP/rescaled.yaml"

grep -m1 -F "is CORRECT" "$TMP/VOLUMETRIC.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/SPECIFIC_ONLY.log"
grep -m1 -F "processor 0 finished normally" "$TMP/VOLUMETRIC.log"

# Nothing in the log flags the units mistake — that is the whole pitfall.
echo "UNITS_WARNINGS=$(grep -ciE 'capa|heat capacity|units' "$TMP/SPECIFIC_ONLY.log")"

# The same centre temperature is reached after a time shorter by exactly rho.
if grep -q "is CORRECT" "$TMP/RESCALED.log"; then
  echo "TIME_CONSTANT_SCALES_WITH_CAPA=yes"
else
  echo "TIME_CONSTANT_SCALES_WITH_CAPA=no"
fi
exit 0
