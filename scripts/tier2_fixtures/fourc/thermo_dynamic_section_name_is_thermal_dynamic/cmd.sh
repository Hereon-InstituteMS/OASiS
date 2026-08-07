#!/bin/bash
# Tier-2 for fourc::thermo#2 — the stand-alone thermal dynamics section is
# 'THERMAL DYNAMIC'.  'THERMO DYNAMIC' — the spelling that matches the
# PROBLEMTYPE, the element category and the discretisation name, so the one
# an agent reaches for — is not a section name on this build.
#
# The same deck twice, differing in that one word.  This is a different code
# path from thermo#1: section names are validated in 4C_io_input_file.cpp
# before any spec matching happens, so the offending name is echoed verbatim
# and nothing else in the deck is even looked at.
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

deck() {  # $1 = dynamics section name
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
$1:
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
RESULT DESCRIPTION:
  - THERMAL:
      DIS: "thermo"
      NODE: 2
      QUANTITY: "temp"
      VALUE: 100.0
      TOLERANCE: 1e-8
YAML
}

deck "THERMAL DYNAMIC" > "$TMP/good.yaml"
deck "THERMO DYNAMIC"  > "$TMP/bad.yaml"

probe THERMAL_DYNAMIC "$TMP/good.yaml"
probe THERMO_DYNAMIC  "$TMP/bad.yaml"

grep -m1 -F "is CORRECT" "$TMP/THERMAL_DYNAMIC.log"
grep -m1 -F "processor 0 finished normally" "$TMP/THERMAL_DYNAMIC.log"
grep -m1 -F "Section 'THERMO DYNAMIC' is not a valid section name." "$TMP/THERMO_DYNAMIC.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/THERMO_DYNAMIC.log"
# It dies at section-name validation, before the discretisation exists:
# no thermo discretisation banner is ever printed for the bad arm.
echo "BAD_ARM_REACHED_DISCRETISATION=$(grep -c 'fill_complete() on discretization thermo' "$TMP/THERMO_DYNAMIC.log")"
# ...and it is NOT the value-level match-tree complaint of thermo#1.
echo "BAD_ARM_MATCHTREE_COMPLAINT=$(grep -c 'Could not match this input' "$TMP/THERMO_DYNAMIC.log")"
exit 0
