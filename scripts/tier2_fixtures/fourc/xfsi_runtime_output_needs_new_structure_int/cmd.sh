#!/bin/bash
# Tier-2 for fourc::fsi_xfem#5 -- you cannot inspect an XFSI fluid VTU in
# ParaView, because 4C refuses to produce one.
#
# Claimed: opening the IO/RUNTIME VTK OUTPUT FLUID VTU in ParaView shows full
#          hexahedral cells crossing the structure.
# Observed: enabling runtime VTK output on the upstream monolithic XFSI deck
#          aborts before the first step with "Runtime output is not available in
#          the old structure time integration!  You need to take the new one,
#          i.e. set INT_STRATEGY: Standard!" from 4C_structure_timint.cpp -- and
#          setting INT_STRATEGY: Standard as instructed does NOT help: the XFSI
#          adapter builds the legacy integrator anyway and the identical throw
#          reappears.  Zero .vtu files are written in either case.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfsi_3D_boxes.4C.yaml) || exit 3
grep -q '  DYNAMICTYPE: "OneStepTheta"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

VTKBLOCK='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/FLUID:
  OUTPUT_FLUID: true
  VELOCITY: true
  PRESSURE: true
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
STRUCTURAL DYNAMIC:
'
python3 - "$BASE" "$TMP" "$VTKBLOCK" <<'PY'
import sys, os
t, TMP, blk = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]
assert "STRUCTURAL DYNAMIC:\n" in t
vtk = t.replace("STRUCTURAL DYNAMIC:\n", blk, 1)
open(os.path.join(TMP, "vtk.yaml"), "w").write(vtk)
open(os.path.join(TMP, "vtkstd.yaml"), "w").write(
    vtk.replace('  DYNAMICTYPE: "OneStepTheta"',
                '  INT_STRATEGY: Standard\n  DYNAMICTYPE: "OneStepTheta"', 1))
PY

mkdir -p "$TMP/a" "$TMP/b" "$TMP/c"
run4c "$BASE"            "$TMP/a/o" > "$TMP/CLEAN.log"  2>&1; echo "EXIT_CLEAN=$?"
run4c "$TMP/vtk.yaml"    "$TMP/b/o" > "$TMP/VTK.log"    2>&1; echo "EXIT_VTK=$?"
run4c "$TMP/vtkstd.yaml" "$TMP/c/o" > "$TMP/VTKSTD.log" 2>&1; echo "EXIT_VTKSTD=$?"

grep -m1 -F "processor 0 finished normally" "$TMP/CLEAN.log"
grep -m1 -F "Runtime output is not available in the old structure time integration!" "$TMP/VTK.log"
grep -m1 -F "4C_structure_timint.cpp" "$TMP/VTK.log"
grep -m1 -F 'INT_STRATEGY: Standard' "$TMP/VTK.log"
# The remedy 4C itself suggests does not work here.
echo "VTKSTD_SAME_THROW=$(grep -c 'Runtime output is not available in the old structure time integration' "$TMP/VTKSTD.log")"
echo "VTK_VTU=$(find "$TMP/b" -name '*.vtu' | wc -l)"
echo "VTKSTD_VTU=$(find "$TMP/c" -name '*.vtu' | wc -l)"
exit 0
