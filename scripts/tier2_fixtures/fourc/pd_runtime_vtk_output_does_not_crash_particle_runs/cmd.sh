#!/bin/bash
# Tier-2 for fourc::particle_pd#12 — FALSE.
#
# Claimed:  "IO/RUNTIME VTK OUTPUT sections are INCOMPATIBLE with particle
#           problems — they crash 4C.  Remove them.  Signal: 4C aborts on
#           startup with `IO/RUNTIME VTK OUTPUT not supported for particle
#           problems` or `RuntimeVTKOutputParams: invalid for PARTICLE`."
# Observed: nothing of the sort.  Bolting IO/RUNTIME VTK OUTPUT and
#           IO/RUNTIME VTK OUTPUT/STRUCTURE onto the upstream 2-D PD deck runs
#           to completion, exits 0, passes all ten of its result tests with
#           abs(diff) exactly 0.00000000000000000e+00 against the untouched
#           baseline, and additionally writes the runtime VTK files.  Neither
#           quoted message exists anywhere in the output.
#
# Deleting these sections on the strength of the entry is therefore a pure loss:
# it removes output an agent may well want and fixes nothing.  Two upstream
# Polymer_Network decks ship IO/RUNTIME VTK OUTPUT blocks for the same reason.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "^IO:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
echo "BASELINE_HAS_RUNTIME_VTK=$(grep -c '^IO/RUNTIME VTK OUTPUT' "$BASE")"

cp "$BASE" "$TMP/novtk.yaml"
python3 - "$BASE" "$TMP/withvtk.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = """IO/RUNTIME VTK OUTPUT:
  OUTPUT_DATA_FORMAT: binary
  INTERVAL_STEPS: 250
  EVERY_ITERATION: false
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
IO:
"""
assert "IO:\n" in t
open(sys.argv[2], "w").write(t.replace("IO:\n", blk, 1))
PY
echo "MUTANT_HAS_RUNTIME_VTK=$(grep -c '^IO/RUNTIME VTK OUTPUT' "$TMP/withvtk.yaml")"

probe NOVTK   "$TMP/novtk.yaml"
probe WITHVTK "$TMP/withvtk.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/NOVTK.log"
echo "NOVTK_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOVTK.log")"

# The section that supposedly crashes 4C does not.
grep -m1 -F "processor 0 finished normally" "$TMP/WITHVTK.log"
echo "WITHVTK_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHVTK.log")"
echo "WITHVTK_EXACT_ZERO_DIFFS=$(grep -c 'is CORRECT, abs(diff)= 0.00000000000000000e+00' "$TMP/WITHVTK.log")"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/WITHVTK.log"

# Neither claimed abort string is anywhere, and nothing warns either.
echo "CLAIMED_NOT_SUPPORTED_TEXT=$(grep -ci 'not supported for particle problems' "$TMP/WITHVTK.log")"
echo "CLAIMED_RUNTIMEVTKOUTPUTPARAMS_TEXT=$(grep -ci 'RuntimeVTKOutputParams' "$TMP/WITHVTK.log")"
echo "WITHVTK_PROC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/WITHVTK.log")"
# And it really did write the extra files.
echo "RUNTIME_VTK_DIR_WRITTEN=$([ -d "$TMP/o_WITHVTK-vtk-files" ] && echo yes || echo no)"

if [ "$(grep -c 'is WRONG --> actresult=' "$TMP/WITHVTK.log")" = "0" ]; then
  echo "VERDICT: RUNTIME_VTK_OUTPUT_CRASHES_PARTICLE_RUNS=no"
else
  echo "VERDICT: RUNTIME_VTK_OUTPUT_CRASHES_PARTICLE_RUNS=yes"
fi
exit 0
