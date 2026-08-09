#!/bin/bash
# Tier-2 for fourc::input_format#4 — a plain 4C run writes NATIVE output, not
# ParaView output.  Looking in the results directory afterwards finds a text
# .control index plus HDF5 .mesh / .result files and not one .vtu or .pvd.
#
# The two ways out are both exercised:
#   (a) ask for runtime output in the deck (IO/RUNTIME VTK OUTPUT + the
#       per-field sub-section), which writes .vtu during the run;
#   (b) run post_vtu --file=<prefix> afterwards on the native files, which
#       converts them.  post_vtu prints a deprecation notice and, without
#       --postprocessor_deprecation_warning_off, waits for Enter -- so it needs
#       its stdin closed in a script.
#
# The HDF5 identity of the native files is checked from their magic bytes, so
# "HDF5-readable but not ParaView-loadable" is not taken on faith.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = extra IO sections (may be empty), $2 = out file
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
IO:
  STRUCT_DISP: true
  OUTPUT_BIN: true
$1
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 2
  MAXTIME: 0.2
  RESULTSEVERY: 1
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
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
YAML
}

RUNTIME_VTK='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true'

mkdir -p "$TMP/native" "$TMP/runtime"
mk ''              "$TMP/native/in.4C.yaml"
mk "$RUNTIME_VTK"  "$TMP/runtime/in.4C.yaml"

run4c "$TMP/native/in.4C.yaml"  "$TMP/native/res"  > "$TMP/native/log"  2>&1
echo "EXIT_NATIVE=$?"
run4c "$TMP/runtime/in.4C.yaml" "$TMP/runtime/res" > "$TMP/runtime/log" 2>&1
echo "EXIT_RUNTIME=$?"

grep -m1 -F "processor 0 finished normally" "$TMP/native/log"
echo "NATIVE_CONTROL=$(ls "$TMP/native" | grep -c '\.control$')"
echo "NATIVE_MESH=$(ls "$TMP/native" | grep -c '\.mesh\.structure\.')"
echo "NATIVE_RESULT=$(ls "$TMP/native" | grep -c '\.result\.structure\.')"
echo "NATIVE_VTU=$(find "$TMP/native" -name '*.vtu' | wc -l)"
echo "NATIVE_PVD=$(find "$TMP/native" -name '*.pvd' | wc -l)"
# The mesh/result files really are HDF5; the control file really is text.
echo "MESH_MAGIC_IS_HDF5=$(head -c 4 "$TMP/native/res.mesh.structure.s0" | od -An -c | grep -c 'H   D   F')"
echo "RESULT_MAGIC_IS_HDF5=$(head -c 4 "$TMP/native/res.result.structure.s0" | od -An -c | grep -c 'H   D   F')"
echo "CONTROL_IS_TEXT=$(head -c 12 "$TMP/native/res.control" | grep -c 'metadata')"

# Route (a): ask for it in the deck.
echo "RUNTIME_VTU=$(find "$TMP/runtime" -name '*.vtu' | wc -l)"
echo "RUNTIME_PVD=$(find "$TMP/runtime" -name '*.pvd' | wc -l)"

# Route (b): convert the native files afterwards.
POST=$(dirname "$BIN")/post_vtu
if [ -x "$POST" ]; then
  ( cd "$TMP/native" && stdbuf -oL -eL "$POST" --file=res > post.log 2>&1 </dev/null )
  echo "EXIT_POST_VTU=$?"
  grep -m1 -F "You are using the post processing functionality of 4C which is deprecated" "$TMP/native/post.log"
  echo "AFTER_POST_VTU_VTU=$(find "$TMP/native" -name '*.vtu' | wc -l)"
  echo "AFTER_POST_VTU_PVD=$(find "$TMP/native" -name '*.pvd' | wc -l)"
else
  echo "FIXTURE_ABORT=no_post_vtu"; exit 3
fi
exit 0
