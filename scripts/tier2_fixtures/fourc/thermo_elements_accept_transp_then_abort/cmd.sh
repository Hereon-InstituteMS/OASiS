#!/bin/bash
# Tier-2 for fourc::thermo#5 — and a FALSIFICATION of how it was worded.
#
# The entry said a wrong eletype in THERMO ELEMENTS "triggers element-spec
# mismatch from core/io/src/4C_io_input_spec_builders.cpp", i.e. that the
# section gates the element category.  It does not.  The element vocabulary
# is global; the section name only says which discretisation the elements go
# into.  So:
#
#   TRANSP without TYPE  -> rejected, but for the MISSING TRANSP KEY:
#                           "Required value 'TYPE' not found in input line"
#   TRANSP with TYPE Std -> ACCEPTED.  The scalar-transport element is built,
#                           put into the 'thermo' discretisation, survives
#                           fill_complete, and the run then dies inside the
#                           thermal time integrator with an UNCAUGHT
#                           Teuchos::Exceptions::InvalidParameterType about
#                           FourC::ScaTra::Action — SIGABRT, exit 134, and no
#                           4C error block at all.
#   THERMO HEX9          -> "Unknown celltype HEX9", and note the file is
#                           4C_fem_general_cell_type_traits.hpp, not the
#                           input-spec builder the entry named.
#
# Nothing anywhere says "this element does not belong in a thermo mesh".
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

# The TYPE Std arm aborts via SIGABRT; do not leave a core file behind.
ulimit -c 0 2>/dev/null || true

deck() {  # $1 = the whole element line
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
  - "$1"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [1.0]
YAML
}

deck "1 THERMO HEX8 1 2 3 4 5 6 7 8 MAT 1"           > "$TMP/good.yaml"
deck "1 TRANSP HEX8 1 2 3 4 5 6 7 8 MAT 1"           > "$TMP/no_type.yaml"
deck "1 TRANSP HEX8 1 2 3 4 5 6 7 8 MAT 1 TYPE Std"  > "$TMP/with_type.yaml"
deck "1 THERMO HEX9 1 2 3 4 5 6 7 8 MAT 1"           > "$TMP/bad_cell.yaml"

probe THERMO         "$TMP/good.yaml"
probe TRANSP_NO_TYPE "$TMP/no_type.yaml"
probe TRANSP_TYPED   "$TMP/with_type.yaml"
probe BAD_CELLTYPE   "$TMP/bad_cell.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/THERMO.log"
grep -m1 -F "Required value 'TYPE' not found in input line" "$TMP/TRANSP_NO_TYPE.log"
grep -m1 -F "Unknown celltype HEX9" "$TMP/BAD_CELLTYPE.log"
grep -m1 -F "4C_fem_general_cell_type_traits.hpp" "$TMP/BAD_CELLTYPE.log"

# The typed TRANSP element is not rejected: it reaches fill_complete on the
# thermo discretisation and only then blows up, unhandled.
grep -m1 -F "fill_complete() on discretization thermo" "$TMP/TRANSP_TYPED.log"
grep -m1 -F "Teuchos::Exceptions::InvalidParameterType" "$TMP/TRANSP_TYPED.log"
grep -m1 -o "FourC::ScaTra::Action" "$TMP/TRANSP_TYPED.log"
if [ "$(grep -c 'fill_complete() on discretization thermo' "$TMP/TRANSP_TYPED.log")" -gt 0 ]; then
  echo "TYPED_TRANSP_REACHED_THERMO_DISCRETISATION=yes"
else
  echo "TYPED_TRANSP_REACHED_THERMO_DISCRETISATION=no"
fi
# No 4C error block: this one escapes the FOUR_C_THROW machinery entirely.
echo "TYPED_TRANSP_FOURC_ERROR_BLOCK=$(grep -c 'PROC 0 ERROR in' "$TMP/TRANSP_TYPED.log")"
# And in NO arm does 4C say the element category is wrong for this section.
echo "ELEMENT_CATEGORY_COMPLAINT=$(cat "$TMP"/TRANSP_NO_TYPE.log "$TMP"/TRANSP_TYPED.log \
  | grep -ciE 'element category|does not belong|not a thermo element|expected element category')"
exit 0
