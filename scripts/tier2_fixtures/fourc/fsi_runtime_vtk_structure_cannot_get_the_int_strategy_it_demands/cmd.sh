#!/bin/bash
# Tier-2 for fourc::fsi#14 — the conflict is real and the entry's diagnosis of
# WHERE it comes from is right, but the message is different and the fix the
# message itself suggests does not work in FSI.
#
# Claimed: "IO/RUNTIME VTK OUTPUT/STRUCTURE may CONFLICT with FSI (INT_STRATEGY
#           override).  Signal: an FSI input with that section aborts with
#           'inconsistent integration strategy' from FSI setup phase; removing
#           the section and using post_vtu after the simulation succeeds.  The
#           override happens inside the FSI adapter, not the user input."
# Observed: the abort is
#             "Runtime output is not available in the old structure time
#              integration! You need to take the new one, i.e. set
#              `INT_STRATEGY: Standard`!"
#           from structure/4C_structure_timint.cpp line 262 — the old time
#           integrator complaining, not an "inconsistent integration strategy".
#           And the last clause of the entry is confirmed the hard way: writing
#           INT_STRATEGY: Standard in STRUCTURAL DYNAMIC, exactly as the message
#           instructs, produces the IDENTICAL abort, because the FSI adapter has
#           already overridden the user's choice.  There is no input the user can
#           write that satisfies it; the section has to go.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^STRUCTURAL DYNAMIC:' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_has_no_structural_dynamic"; exit 3; }
grep -q 'IO/RUNTIME VTK OUTPUT' "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_already_requests_runtime_vtk"; exit 3; }

# The pathology: ask for runtime VTK structure output in an FSI input.
REQUEST_RUNTIME_VTK_STRUCTURE=yes

cp "$BASE" "$TMP/novtk.yaml"
python3 - "$BASE" "$TMP" "$REQUEST_RUNTIME_VTK_STRUCTURE" <<'PY'
import sys
src, tmp, do = sys.argv[1:4]
t = open(src).read()
block = ("IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n"
         "IO/RUNTIME VTK OUTPUT/STRUCTURE:\n  OUTPUT_STRUCTURE: true\n"
         "  DISPLACEMENT: true\n")
assert "PROBLEM TYPE:" in t and "STRUCTURAL DYNAMIC:\n  PREDICT:" in t
add = block if do == "yes" else ""
open(tmp + "/vtk_default.yaml", "w").write(t.replace("PROBLEM TYPE:", add + "PROBLEM TYPE:", 1))
u = t.replace("STRUCTURAL DYNAMIC:\n  PREDICT:",
              'STRUCTURAL DYNAMIC:\n  INT_STRATEGY: "Standard"\n  PREDICT:', 1)
open(tmp + "/vtk_standard.yaml", "w").write(u.replace("PROBLEM TYPE:", add + "PROBLEM TYPE:", 1))
PY
echo "VTK_ARMS_REQUEST_STRUCTURE_OUTPUT=$(grep -c 'IO/RUNTIME VTK OUTPUT/STRUCTURE:' "$TMP/vtk_default.yaml")"
echo "STANDARD_ARM_SETS_INT_STRATEGY=$(grep -c 'INT_STRATEGY: "Standard"' "$TMP/vtk_standard.yaml")"

probe NOVTK       "$TMP/novtk.yaml"
probe VTKDEFAULT  "$TMP/vtk_default.yaml"
probe VTKSTANDARD "$TMP/vtk_standard.yaml"

# Control: without the section the same deck runs.
grep -m1 -F "processor 0 finished normally" "$TMP/NOVTK.log"
grep -m1 -F "OK (6)" "$TMP/NOVTK.log"

# The real message, and the file it comes from.
grep -m1 -F "Runtime output is not available in the old structure time integration!" "$TMP/VTKDEFAULT.log"
grep -m1 -F "INT_STRATEGY: Standard" "$TMP/VTKDEFAULT.log"
grep -m1 -F "4C_structure_timint.cpp" "$TMP/VTKDEFAULT.log"
echo "VTKDEFAULT_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/VTKDEFAULT.log")"

# Following the message's own instruction does not help: the FSI adapter has
# already chosen the integrator.
grep -m1 -F "Runtime output is not available in the old structure time integration!" "$TMP/VTKSTANDARD.log"
grep -m1 -F "4C_structure_timint.cpp" "$TMP/VTKSTANDARD.log"
echo "VTKSTANDARD_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/VTKSTANDARD.log")"
echo "SETTING_INT_STRATEGY_HELPS=$([ "$(grep -c 'PROC 0 ERROR' "$TMP/VTKSTANDARD.log")" -eq 0 ] && echo yes || echo no)"
echo "CLAIMED_INCONSISTENT_STRATEGY_TEXT=$(cat "$TMP"/VTKDEFAULT.log "$TMP"/VTKSTANDARD.log \
      | grep -ci 'inconsistent integration strategy')"
exit 0
