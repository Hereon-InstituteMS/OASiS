#!/bin/bash
# Tier-2 for fourc::thermal#1 — THERMAL and THERMO are not interchangeable and
# there is no rule of thumb.  One working standalone Thermo deck, six copies,
# each swapping THERMAL<->THERMO in exactly one place.  Every swap fails, and
# each fails DIFFERENTLY — which is the point: you cannot recognise the mistake
# from the error, you have to know the table.
#
#   DESIGN SURF THERMAL DIRICH CONDITIONS -> not a valid section name  (input_file.cpp)
#   THERMO DYNAMIC                        -> not a valid section name  (input_file.cpp)
#   THERMAL ELEMENTS                      -> not a valid section name  (input_file.cpp)
#   RESULT DESCRIPTION  - THERMO:         -> Could not match this input (input_spec_builders.cpp)
#   DIS: "thermal"                        -> expected 1 tests but performed 0 (utils_result_test.cpp)
#   element type THERMAL                  -> Unknown type 'THERMAL' of finite element
#
# Contrast with fourc::thermal#0: the THERMAL-prefixed condition section fails
# LOUDLY; the THERMO-prefixed one is accepted and silently dropped.
#
# The claim previously attributed the section-name message to
# input_spec_builders.cpp and quoted it as "unknown section".  Neither holds:
# the message comes from core/io/src/4C_io_input_file.cpp and the string
# "unknown section" is absent.  Both are pinned below.
. "$(dirname "$0")/../_lib/preamble.sh"

cat > "$TMP/base.yaml" <<'YAML'
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
  NAME: "Thermal_Solver"
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

swap() {  # $1 = label, $2 = search, $3 = replace
  python3 - "$TMP/base.yaml" "$TMP/$1.yaml" "$2" "$3" <<'PY'
import sys
src, dst, a, b = sys.argv[1:5]
t = open(src).read()
if a not in t:
    print("FIXTURE_ABORT=inline_deck_changed"); sys.exit(3)
open(dst, "w").write(t.replace(a, b))
PY
}

swap dirich  "DESIGN SURF DIRICH CONDITIONS" "DESIGN SURF THERMAL DIRICH CONDITIONS"
swap dyn     "THERMAL DYNAMIC:"              "THERMO DYNAMIC:"
swap elesec  "THERMO ELEMENTS:"              "THERMAL ELEMENTS:"
swap rdgroup "  - THERMAL:"                  "  - THERMO:"
swap dis     'DIS: "thermo"'                 'DIS: "thermal"'
swap eletype '"1 THERMO HEX8'                '"1 THERMAL HEX8'

probe BASE    "$TMP/base.yaml"
probe DIRICH  "$TMP/dirich.yaml"
probe DYN     "$TMP/dyn.yaml"
probe ELESEC  "$TMP/elesec.yaml"
probe RDGROUP "$TMP/rdgroup.yaml"
probe DIS     "$TMP/dis.yaml"
probe ELETYPE "$TMP/eletype.yaml"

grep -m1 -F "is CORRECT" "$TMP/BASE.log"
grep -m1 -F "Section 'DESIGN SURF THERMAL DIRICH CONDITIONS' is not a valid section name." "$TMP/DIRICH.log"
grep -m1 -oF "4C_io_input_file.cpp" "$TMP/DIRICH.log"
grep -m1 -F "Section 'THERMO DYNAMIC' is not a valid section name." "$TMP/DYN.log"
grep -m1 -F "Section 'THERMAL ELEMENTS' is not a valid section name." "$TMP/ELESEC.log"
grep -m1 -F "Could not match this input" "$TMP/RDGROUP.log"
grep -m1 -F "expected 1 tests but performed 0" "$TMP/DIS.log"
grep -m1 -F "Unknown type 'THERMAL' of finite element" "$TMP/ELETYPE.log"

# The previously catalogued wording does not exist.
echo "CLAIMED_UNKNOWN_SECTION_TEXT=$(grep -ci 'unknown section' "$TMP/DIRICH.log")"
echo "CLAIMED_SPEC_BUILDERS_ORIGIN=$(grep -ci 'input_spec_builders' "$TMP/DIRICH.log")"
exit 0
