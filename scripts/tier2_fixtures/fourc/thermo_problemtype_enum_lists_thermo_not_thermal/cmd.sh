#!/bin/bash
# Tier-2 for fourc::thermo#1 — PROBLEMTYPE is 'Thermo', not 'Thermal'.
#
# The same one-element heat-conduction deck twice, differing only in that
# token.  'Thermo' runs to completion; 'Thermal' is rejected by the
# InputSpec match tree, which prints the WHOLE enum — and that printed
# list is the useful part, because it is the only place an agent can read
# off the accepted spelling.  The fixture asserts the list contains
# '|Thermo|' and does NOT contain '|Thermal|'.
#
# Note the failure is NOT a section-name error (4C_io_input_file.cpp):
# 'PROBLEM TYPE' is a perfectly good section, so the rejection happens one
# level down, in 4C_io_input_spec_builders.cpp, against the value.
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

deck() {  # $1 = PROBLEMTYPE token
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "$1"
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "T"
DESIGN SURF DIRICH CONDITIONS:
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
YAML
}

deck "Thermo"  > "$TMP/good.yaml"
deck "Thermal" > "$TMP/bad.yaml"

probe THERMO  "$TMP/good.yaml"
probe THERMAL "$TMP/bad.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/THERMO.log"
grep -m1 -F "Could not match this input" "$TMP/THERMAL.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/THERMAL.log"
grep -m1 -o "Candidate deprecated_selection 'PROBLEMTYPE' has wrong value, possible values:" \
     "$TMP/THERMAL.log"

# The enumerated list is the payload: it carries the accepted spelling and
# proves the rejected one is simply not a member.
if grep -q "|Thermo|" "$TMP/THERMAL.log"; then
  echo "ENUM_LISTS_THERMO=yes"
else
  echo "ENUM_LISTS_THERMO=no"
fi
if grep -q "|Thermal|" "$TMP/THERMAL.log"; then
  echo "ENUM_LISTS_THERMAL=yes"
else
  echo "ENUM_LISTS_THERMAL=no"
fi
# It is a value rejection, not a section-name rejection.
echo "SECTION_NAME_COMPLAINT=$(grep -c 'is not a valid section name' "$TMP/THERMAL.log")"
exit 0
