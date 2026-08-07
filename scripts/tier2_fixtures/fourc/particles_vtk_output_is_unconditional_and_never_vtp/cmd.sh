#!/bin/bash
# Tier-2 for fourc::particles#1 — falsified on every clause.
#
# Claimed:  "IO/RUNTIME VTK OUTPUT/PARTICLES must be added for ParaView output
#           (VTP files).  Signal: a PD simulation runs to completion but no
#           .vtp / .pvd files are in the output directory — 4C produces native
#           files only, no particle output unless the PARTICLES subsection is
#           configured with PARTICLE_OUTPUT: true."
# Observed: upstream particle_sph_2d_pdbody_gravity.4C.yaml has no IO/RUNTIME
#           VTK OUTPUT section of any kind, and 4C writes ParaView particle
#           output anyway: one .pvd per phase plus a <prefix>-vtk-files
#           directory of .vtu/.pvtu pieces.  Nothing has to be switched on.
#           The file type is wrong too — 4C emits no .vtp at all.
#           And the section the entry tells you to add does not exist: adding
#           IO/RUNTIME VTK OUTPUT/PARTICLES with PARTICLE_OUTPUT: true is a hard
#           parse error, "Section 'IO/RUNTIME VTK OUTPUT/PARTICLES' is not a
#           valid section name." from core/io/src/4C_io_input_file.cpp, so
#           following the advice turns a working deck into a dead one.
#           4C's whole vocabulary here is IO/RUNTIME VTK OUTPUT plus /BEAMS,
#           /FLUID and /STRUCTURE; there is no /PARTICLES and no PARTICLE_OUTPUT
#           key anywhere in the source.
#
# Particle output frequency is governed by RESULTSEVERY in PARTICLE DYNAMIC.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q '^IO/RUNTIME VTK OUTPUT' "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP/asis.yaml" "$TMP/claimed.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "BINNING STRATEGY:"
assert anchor in t, "upstream deck no longer carries a BINNING STRATEGY section"
blk = '''IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 250
IO/RUNTIME VTK OUTPUT/PARTICLES:
  PARTICLE_OUTPUT: true
'''
i = t.index(anchor)
open(sys.argv[2], "w").write(t)
open(sys.argv[3], "w").write(t[:i] + blk + t[i:])
PY
echo "ASIS_RUNTIME_VTK_SECTIONS=$(grep -c '^IO/RUNTIME VTK OUTPUT' "$TMP/asis.yaml")"
echo "CLAIMED_RUNTIME_VTK_SECTIONS=$(grep -c '^IO/RUNTIME VTK OUTPUT' "$TMP/claimed.yaml")"

probe ASIS    "$TMP/asis.yaml"
probe CLAIMED "$TMP/claimed.yaml"

# The untouched deck runs and writes ParaView particle output with no section.
grep -m1 -F "processor 0 finished normally" "$TMP/ASIS.log"
echo "ASIS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ASIS.log")"
echo "PVD_FILES_WITHOUT_ANY_VTK_SECTION=$(ls "$TMP"/o_ASIS*.pvd 2>/dev/null | wc -l)"
echo "VTU_FILES_WITHOUT_ANY_VTK_SECTION=$(ls "$TMP"/o_ASIS-vtk-files/*.vtu 2>/dev/null | wc -l)"
echo "VTP_FILES_ANYWHERE=$(find "$TMP" -name '*.vtp' 2>/dev/null | wc -l)"
ls "$TMP"/o_ASIS-vtk-files/ 2>/dev/null | grep -m1 -oE 'particle-pdphase-owned-00000-0\.vtu'

# Following the entry's advice kills the deck.
grep -m1 -F "Section 'IO/RUNTIME VTK OUTPUT/PARTICLES' is not a valid section name." "$TMP/CLAIMED.log"
grep -m1 -oE "4C_io_input_file\.cpp, line [0-9]+" "$TMP/CLAIMED.log"
echo "CLAIMED_TIME_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/CLAIMED.log")"
exit 0
