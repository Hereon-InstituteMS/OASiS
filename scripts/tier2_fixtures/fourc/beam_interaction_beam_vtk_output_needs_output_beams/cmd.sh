#!/bin/bash
# Tier-2 for fourc::beam_interaction#6 — a beam run with OUTPUT_BEAMS off writes
# NO visualisation output at all, exits 0, and says nothing.
#
# Claimed: "shows structure-*.vtu files but no beams-*.vtu".
# Observed, on upstream beam_runtime_ghosting_output (a beams-only mesh, all
# BEAM3R LINE2):
#   * OUTPUT_BEAMS: true  -> three files named structure-beams-NNNNN-0.vtu plus
#     res-structure-beams.pvd. The beam output is NOT called beams-*.vtu; it is
#     called structure-beams-*.
#   * OUTPUT_BEAMS: false -> ZERO .vtu and ZERO .pvd, even though
#     IO/RUNTIME VTK OUTPUT/STRUCTURE sets OUTPUT_STRUCTURE: true. On a
#     discretisation whose elements are all beams the structure writer emits
#     nothing, so there are no structure-*.vtu files to notice the absence
#     against. Both runs exit 0 with no warning.
#
# Each arm runs in its own directory because 4C writes its VTU tree relative to
# the working directory.
. "$(dirname "$0")/../_lib/preamble.sh"

DECK=beam_runtime_ghosting_output
BASE=$(upstream "$DECK.4C.yaml") || exit 3
XML=$(upstream "$DECK.xml")      || exit 3
cd "$TMP" || exit 3
cp "$XML" .
cp "$BASE" on.yaml
grep -q "  OUTPUT_BEAMS: true" on.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  OUTPUT_STRUCTURE: true" on.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
sed 's/  OUTPUT_BEAMS: true/  OUTPUT_BEAMS: false/' on.yaml > off.yaml

mkdir -p d_on d_off
( cd d_on  && stdbuf -oL -eL "$BIN" ../on.yaml  res > run.log 2>&1; echo "EXIT_ON=$?" )
( cd d_off && stdbuf -oL -eL "$BIN" ../off.yaml res > run.log 2>&1; echo "EXIT_OFF=$?" )

grep -m1 -F "processor 0 finished normally" d_on/run.log
echo "ON_BEAM_VTU=$(find d_on  -name 'structure-beams-*.vtu' | wc -l)"
echo "OFF_BEAM_VTU=$(find d_off -name 'structure-beams-*.vtu' | wc -l)"
echo "ON_ANY_VTU=$(find d_on  -name '*.vtu' | wc -l)"
echo "OFF_ANY_VTU=$(find d_off -name '*.vtu' | wc -l)"
echo "OFF_ANY_PVD=$(find d_off -name '*.pvd' | wc -l)"
# A beams-only mesh produces no plain structure-*.vtu even with the beams on.
echo "ON_PLAIN_STRUCTURE_VTU=$(find d_on -name 'structure-[0-9]*.vtu' | wc -l)"
# ...and the silent arm says nothing about missing output.
echo "OFF_OUTPUT_WARNINGS=$(grep -ciE 'no output|nothing to write|skipping output' d_off/run.log)"
exit 0
