#!/bin/bash
# Tier-2 for fourc::thermo#4 — a PROBLEMTYPE 'Thermo' run builds ONE
# discretisation.  Bolting a STRUCTURE ELEMENTS block (and a structural
# material) onto the deck does not create a second field, does not couple
# anything, and does not produce a single word of complaint: the run exits 0
# having solved heat conduction alone.
#
# Three arms:
#   THERMO_ONLY          plain thermo deck
#   THERMO_PLUS_STRUCT   same deck + STRUCTURE ELEMENTS + MAT_Struct_...
#   TSI                  upstream tsi_heatflux_monolithic.4C.yaml
#
# The contrast is the point: the TSI arm prints a 'discretization structure'
# banner, the hand-built one never does, and there is no diagnostic to tell
# the two apart from inside the thermo run.
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
DECKS=""
for _d in "${FOURC_INPUT_FILES:-}" "$HOME/4C/tests/input_files" \
          "$HOME/Schreibtisch/4C-src/4C/tests/input_files"; do
  [ -n "${_d:-}" ] && [ -d "$_d" ] && DECKS="$_d" && break
done
upstream() {  # <name.4C.yaml> -> its path, or abort loudly
  if [ -z "$DECKS" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (set FOURC_INPUT_FILES)"; exit 3; fi
  if [ ! -f "$DECKS/$1" ]; then
    echo "FIXTURE_ABORT=no_upstream_decks (missing $1)"; exit 3; fi
  printf '%s\n' "$DECKS/$1"
}
# ---------------------------------------------------------------------------

deck() {  # $1 = extra sections appended verbatim
cat <<YAML
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
  - MAT: 2
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - THERMAL:
      DIS: "thermo"
      NODE: 2
      QUANTITY: "temp"
      VALUE: 100.0
      TOLERANCE: 1e-8
$1
YAML
}

STRUCT_BLOCK='STRUCTURE ELEMENTS:
  - "2 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 2 KINEM nonlinear"'

deck ""              > "$TMP/thermo_only.yaml"
deck "$STRUCT_BLOCK" > "$TMP/thermo_plus_struct.yaml"

# The pathology has to actually be in the deck for the rest to mean anything.
echo "DECK_CARRIES_STRUCTURE_ELEMENTS=$(grep -c '^STRUCTURE ELEMENTS:' "$TMP/thermo_plus_struct.yaml")"

TSI=$(upstream tsi_heatflux_monolithic.4C.yaml) || exit 3

probe THERMO_ONLY        "$TMP/thermo_only.yaml"
probe THERMO_PLUS_STRUCT "$TMP/thermo_plus_struct.yaml"
probe TSI                "$TSI"

grep -m1 -F "fill_complete() on discretization thermo" "$TMP/THERMO_PLUS_STRUCT.log"
grep -m1 -F "fill_complete() on discretization structure" "$TMP/TSI.log"
grep -m1 -F "processor 0 finished normally" "$TMP/THERMO_PLUS_STRUCT.log"
grep -m1 -F "is CORRECT" "$TMP/THERMO_PLUS_STRUCT.log"

echo "STRUCT_DISCRETISATION_IN_HANDBUILT=$(grep -c 'fill_complete() on discretization structure' "$TMP/THERMO_PLUS_STRUCT.log")"
if [ "$(grep -c 'fill_complete() on discretization structure' "$TMP/TSI.log")" -gt 0 ]; then
  echo "STRUCT_DISCRETISATION_IN_TSI=yes"
else
  echo "STRUCT_DISCRETISATION_IN_TSI=no"
fi
# Not one word about the block that was thrown away.
echo "IGNORED_BLOCK_WARNINGS=$(grep -ciE 'structure.*(ignor|unus|not read|no effect)|unused section' "$TMP/THERMO_PLUS_STRUCT.log")"
exit 0
