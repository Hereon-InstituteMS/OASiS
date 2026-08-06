#!/bin/bash
# Tier-2 for fourc::thermal#5 — thermal runtime VTU output.  Five arms of the
# same standalone Thermo deck, NUMSTEP 4:
#
#   NONE      no VTK section at all              -> exit 0, 0 .vtu, only .control/.mesh/.result
#   BOTH      IO/RUNTIME VTK OUTPUT + thermal    -> exit 0, 5 .vtu
#   THERMAL   only THERMAL DYNAMIC/RUNTIME VTK   -> exit 0, 5 .vtu   <- "needs BOTH" is false
#   IO_ONLY   only IO/RUNTIME VTK OUTPUT         -> exit 0, 0 .vtu
#   INTERVAL  INTERVAL_STEPS inside the thermal subsection -> exit 1, fatal
#
# Two halves of the claim hold and two do not.
#
#   HOLDS  the thermal subsection's key list is exactly OUTPUT_THERMO,
#          TEMPERATURE, TEMPERATURE_RATE, CONDUCTIVITY, ELEMENT_OWNER,
#          ELEMENT_GID, NODE_GID and carries NO frequency key, so
#          INTERVAL_STEPS there is fatal, not ignored:
#          'Could not match this input' echoing the block.
#   HOLDS  with no VTK section the run still succeeds and writes only the
#          binary .control / .mesh / .result files.
#   FALSE  "needs BOTH sections": the thermal subsection ALONE writes .vtu.
#          The parent section is not required, and IO_ONLY writes nothing.
#   FALSE  "INTERVAL_STEPS lives in the parent IO/RUNTIME VTK OUTPUT" — it
#          parses there but does not throttle the thermo writer at all
#          (INTERVAL_STEPS 2 still gives 5 files).  The frequency knob that
#          does work is THERMAL DYNAMIC/RESULTSEVERY (2 -> 3 files).
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = sections prepended, $2 = extra THERMAL DYNAMIC keys
cat <<YAML
$1PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  NUMSTEP: 4
  MAXTIME: 4
  LINEAR_SOLVER: 1
$2SOLVER 1:
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
}

TH='THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
'
IO='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
'
IO2='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 2
'
BAD='THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
  INTERVAL_STEPS: 1
'

deck ""            ""                  > "$TMP/none.yaml"
deck "$IO$TH"      ""                  > "$TMP/both.yaml"
deck "$TH"         ""                  > "$TMP/thermal.yaml"
deck "$IO"         ""                  > "$TMP/io_only.yaml"
deck "$BAD"        ""                  > "$TMP/interval.yaml"
deck "$IO2$TH"     ""                  > "$TMP/io_every2.yaml"
deck "$TH"         "  RESULTSEVERY: 2
"                                      > "$TMP/resultsevery2.yaml"

count_vtu() { ls "$TMP"/"$1"-vtk-files/*.vtu 2>/dev/null | wc -l; }
run_arm() {  # $1 = label, $2 = deck
  ( cd "$TMP" && stdbuf -oL -eL "$BIN" "$2" "$TMP/$1" > "$TMP/$1.log" 2>&1 )
  echo "EXIT_$1=$?"
}

run_arm NONE      "$TMP/none.yaml"
run_arm BOTH      "$TMP/both.yaml"
run_arm THERMAL   "$TMP/thermal.yaml"
run_arm IO_ONLY   "$TMP/io_only.yaml"
run_arm INTERVAL  "$TMP/interval.yaml"
run_arm IO_EVERY2 "$TMP/io_every2.yaml"
run_arm RESEVERY2 "$TMP/resultsevery2.yaml"

echo "VTU_NONE=$(count_vtu NONE)"
echo "VTU_BOTH=$(count_vtu BOTH)"
echo "VTU_THERMAL_SUBSECTION_ALONE=$(count_vtu THERMAL)"
echo "VTU_IO_SECTION_ALONE=$(count_vtu IO_ONLY)"
echo "VTU_IO_INTERVAL_STEPS_2=$(count_vtu IO_EVERY2)"
echo "VTU_THERMAL_RESULTSEVERY_2=$(count_vtu RESEVERY2)"
# With no VTK section the binary result files are still written.
echo "BINARY_FILES_NONE=$(ls "$TMP"/NONE.control "$TMP"/NONE.mesh.thermo.s0 "$TMP"/NONE.result.thermo.s1 2>/dev/null | wc -l)"
# INTERVAL_STEPS inside the thermal subsection is fatal, and the block is echoed.
grep -m1 -F "Could not match this input" "$TMP/INTERVAL.log"
grep -m1 -F "THERMAL DYNAMIC/RUNTIME VTK OUTPUT:" "$TMP/INTERVAL.log"
grep -m1 -oF "4C_io_input_spec_builders.cpp" "$TMP/INTERVAL.log"
grep -m1 -F "is CORRECT" "$TMP/THERMAL.log"
exit 0
