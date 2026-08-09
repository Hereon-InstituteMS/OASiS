#!/bin/bash
# Tier-2 for fourc::fbi#4 — configuring IO/RUNTIME VTK OUTPUT/STRUCTURE on an FBI
# problem does not give you "empty output".  It gives you NO FILE AT ALL, and the
# run exits 0 with every result test passing, so nothing tells you.
#
# Upstream fbi_mortar_solidcoupling.4C.yaml asks for IO/RUNTIME VTK OUTPUT/BEAMS
# and writes, per time step, structure-beams-NNNNN-0.vtu + .pvtu plus a
# <prefix>-structure-beams.pvd collection.
#
# Replace that subsection with IO/RUNTIME VTK OUTPUT/STRUCTURE (OUTPUT_STRUCTURE:
# true, DISPLACEMENT: true) and the beam files disappear with no replacement:
# zero structure-*.vtu, zero *-structure.pvd.  The beam-to-fluid coupling output
# still appears, which is exactly what makes it look like output "worked".
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fbi_mortar_solidcoupling.4C.yaml) || exit 3
grep -q '^IO/RUNTIME VTK OUTPUT/BEAMS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_beams_output_section_changed"; exit 3; }
grep -q '  OUTPUT_BEAMS: true' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_output_beams_key_changed"; exit 3; }
cp "$(dirname "$BASE")/beam_flow_solver.xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_nox_xml"; exit 3; }
cd "$TMP" || exit 3

# The pathology: which runtime-VTK subsection the second arm asks for.
WRONG_OUTPUT_SECTION='IO/RUNTIME VTK OUTPUT/STRUCTURE:\n  OUTPUT_STRUCTURE: true\n  DISPLACEMENT: true\n'

cp "$BASE" "$TMP/beams.yaml"
python3 - "$BASE" "$TMP/structure.yaml" "$WRONG_OUTPUT_SECTION" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
new = sys.argv[3].encode().decode("unicode_escape")
m = re.search(r'^IO/RUNTIME VTK OUTPUT/BEAMS:\n(  \S.*\n)+', t, re.M)
assert m, "upstream deck no longer carries an IO/RUNTIME VTK OUTPUT/BEAMS block"
open(sys.argv[2], "w").write(t[:m.start()] + new + t[m.end():])
PY

probe BEAMS     "$TMP/beams.yaml"
probe STRUCTURE "$TMP/structure.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BEAMS.log"
grep -m1 -F "processor 0 finished normally" "$TMP/STRUCTURE.log"
grep -m1 -F "OK (6)" "$TMP/STRUCTURE.log"

count() { ls "$1" 2>/dev/null | grep -c "$2"; }
echo "BEAMS_ARM_BEAM_VTU=$(count "$TMP/o_BEAMS-vtk-files" '^structure-beams-.*\.vtu$')"
echo "STRUCTURE_ARM_BEAM_VTU=$(count "$TMP/o_STRUCTURE-vtk-files" '^structure-beams-.*\.vtu$')"
echo "STRUCTURE_ARM_ANY_STRUCTURE_VTU=$(count "$TMP/o_STRUCTURE-vtk-files" '^structure')"
echo "BEAMS_ARM_PVD=$(ls "$TMP" | grep -c '^o_BEAMS-.*\.pvd$')"
echo "STRUCTURE_ARM_PVD=$(ls "$TMP" | grep -c '^o_STRUCTURE-.*\.pvd$')"
echo "STRUCTURE_ARM_COUPLING_VTU=$(count "$TMP/o_STRUCTURE-vtk-files" '^beam-to-fluid-.*\.vtu$')"
# Nothing in the log points out that the requested output was not produced.
echo "STRUCTURE_OUTPUT_WARNINGS=$(grep -ciE 'output.*(ignor|not written|no dofs|empty)|no structure output' "$TMP/STRUCTURE.log")"
echo "STRUCTURE_ARM_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/STRUCTURE.log")"
exit 0
