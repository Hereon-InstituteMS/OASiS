#!/bin/bash
# Tier-2 for fourc::input_format#3 — section names are matched against a closed
# list and an unrecognised one is fatal, with a message that names the offending
# section: "Section '<name>' is not a valid section name." from
# core/io/src/4C_io_input_file.cpp.  A name that IS valid but carries a key the
# section does not own fails somewhere else entirely, with "Could not match this
# input" from 4C_io_input_spec_builders.cpp and the offending YAML echoed.
#
# The two failures are worth telling apart: the first means "no such section",
# the second means "right section, wrong contents".
#
# The authority for the list is the binary itself.  This fixture reads it out of
# `4C --parameters`, where sections appear as `    - name: <NAME>` under the
# top-level `sections:` key, and confirms the three names the catalogue calls
# out: DESIGN LINE THERMO DIRICH CONDITIONS exists, the transposed spelling
# DESIGN THERMO LINE DIRICH CONDITIONS does not, and LIFT&DRAG is a SURF
# condition only -- there is no line version.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = extra section text (may be empty), $2 = out file
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 1
  MAXTIME: 0.1
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
$1
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 1, 0]
    FUNCT: [0, 0, 0]
    TYPE: "Live"
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
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: 4.47909266337460053e-03
      TOLERANCE: 1e-12
YAML
}

mk '' "$TMP/good.yaml"
mk 'EVERY_ITERATION:
  OUTPUT: true' "$TMP/everyiter.yaml"
mk 'DESIGN THERMO LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0]
    FUNCT: [0]' "$TMP/transposed.yaml"
mk 'DESIGN FLUID LINE LIFT&DRAG:
  - E: 1
    LABEL: 1' "$TMP/liftdrag.yaml"
# A section that DOES exist, holding a key it does not own.
mk 'IO:
  BANANA: 3' "$TMP/badkey.yaml"

probe GOOD       "$TMP/good.yaml"
probe EVERYITER  "$TMP/everyiter.yaml"
probe TRANSPOSED "$TMP/transposed.yaml"
probe LIFTDRAG   "$TMP/liftdrag.yaml"
probe BADKEY     "$TMP/badkey.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
grep -m1 -F "Section 'EVERY_ITERATION' is not a valid section name." "$TMP/EVERYITER.log"
grep -m1 -F "Section 'DESIGN THERMO LINE DIRICH CONDITIONS' is not a valid section name." "$TMP/TRANSPOSED.log"
grep -m1 -F "Section 'DESIGN FLUID LINE LIFT&DRAG' is not a valid section name." "$TMP/LIFTDRAG.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/EVERYITER.log"
# A real section with a key it does not own fails from a different place.
grep -m1 -F "Could not match this input" "$TMP/BADKEY.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/BADKEY.log"
echo "BADKEY_BLAMED_THE_SECTION_NAME=$(grep -c 'is not a valid section name' "$TMP/BADKEY.log")"

# The binary's own list is the authority; ask it rather than a catalogue.
"$BIN" --parameters 2>/dev/null > "$TMP/params.yaml"
sec() { awk '/^sections:/{f=1;next} /^[A-Za-z_$]/{f=0} f' "$TMP/params.yaml" | grep -c "^    - name: $1\$"; }
echo "SCHEMA_HAS_DESIGN_LINE_THERMO_DIRICH=$(sec 'DESIGN LINE THERMO DIRICH CONDITIONS')"
echo "SCHEMA_HAS_DESIGN_THERMO_LINE_DIRICH=$(sec 'DESIGN THERMO LINE DIRICH CONDITIONS')"
echo "SCHEMA_HAS_FLUID_SURF_LIFTDRAG=$(sec 'DESIGN FLUID SURF LIFT&DRAG')"
echo "SCHEMA_HAS_FLUID_LINE_LIFTDRAG=$(sec 'DESIGN FLUID LINE LIFT&DRAG')"
echo "SCHEMA_HAS_EVERY_ITERATION=$(sec 'EVERY_ITERATION')"
echo "SCHEMA_DESIGN_SECTION_COUNT=$(awk '/^sections:/{f=1;next} /^[A-Za-z_$]/{f=0} f' "$TMP/params.yaml" | grep -c '^    - name: DESIGN ')"
exit 0
