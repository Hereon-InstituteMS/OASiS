#!/bin/bash
# Tier-2 for fourc::fsi#18 — the subsection really does not exist, but 4C rejects
# it as a SECTION NAME at file level, not as an "unknown subsection".
#
# Claimed: "IO/RUNTIME VTK OUTPUT/ALE does NOT exist — it crashes 4C.  Signal:
#           writing /ALE as a subsection causes an immediate parse failure with
#           'unknown subsection ALE in IO/RUNTIME VTK OUTPUT' from
#           4C_io_input_spec_builders.cpp.  Only /STRUCTURE and /FLUID
#           subsections are valid for FSI VTK output."
# Observed: the abort is
#             "Section 'IO/RUNTIME VTK OUTPUT/ALE' is not a valid section name."
#           from core/io/src/4C_io_input_file.cpp line 546 — the same check that
#           rejects any misspelled top-level section, quoting the WHOLE path as
#           one name.  The claimed sentence and the claimed source file are
#           both absent.  This matters for recovery: there is no candidate list
#           and no "did you mean", so nothing in the output tells the reader that
#           /FLUID would have worked.  The control arm shows it does: the same
#           deck with IO/RUNTIME VTK OUTPUT/FLUID runs to completion.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q 'IO/RUNTIME VTK OUTPUT' "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_already_requests_runtime_vtk"; exit 3; }

# The pathology: request runtime VTK output for the ALE field.
VTK_SUBSECTION=ALE

cp "$BASE" "$TMP/plain.yaml"
python3 - "$BASE" "$TMP" "$VTK_SUBSECTION" <<'PY'
import sys
src, tmp, sub = sys.argv[1:4]
t = open(src).read()
assert "PROBLEM TYPE:" in t
def deck(name, body):
    return t.replace("PROBLEM TYPE:",
                     "IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n"
                     "IO/RUNTIME VTK OUTPUT/%s:\n%sPROBLEM TYPE:" % (name, body), 1)
open(tmp + "/subsection.yaml", "w").write(deck(sub, "  OUTPUT_%s: true\n" % sub))
open(tmp + "/fluid.yaml", "w").write(
    deck("FLUID", "  OUTPUT_FLUID: true\n  VELOCITY: true\n"))
PY
echo "BAD_ARM_SUBSECTION=$(grep -o 'IO/RUNTIME VTK OUTPUT/[A-Z]*' "$TMP/subsection.yaml" | tail -1 | awk -F/ '{print $NF}')"

probe PLAIN      "$TMP/plain.yaml"
probe SUBSECTION "$TMP/subsection.yaml"
probe FLUID      "$TMP/fluid.yaml"

grep -m1 -F "OK (6)" "$TMP/PLAIN.log"

# The real rejection.
grep -m1 -F "Section 'IO/RUNTIME VTK OUTPUT/ALE' is not a valid section name." "$TMP/SUBSECTION.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/SUBSECTION.log"
echo "SUBSECTION_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/SUBSECTION.log")"
echo "SUBSECTION_CLAIMED_SENTENCE=$(grep -ci 'unknown subsection' "$TMP/SUBSECTION.log")"
echo "SUBSECTION_CLAIMED_SOURCE_FILE=$(grep -ci '4C_io_input_spec_builders.cpp' "$TMP/SUBSECTION.log")"
# No candidate list, no suggestion: the reader is told nothing about /FLUID.
echo "SUBSECTION_OFFERS_A_CANDIDATE_LIST=$(grep -ci 'against the given input specification' "$TMP/SUBSECTION.log")"
echo "SUBSECTION_MENTIONS_FLUID_OR_STRUCTURE=$(grep -ci 'RUNTIME VTK OUTPUT/\(FLUID\|STRUCTURE\)' "$TMP/SUBSECTION.log")"

# ...and /FLUID really is the working spelling in the same place.
grep -m1 -F "processor 0 finished normally" "$TMP/FLUID.log"
grep -m1 -F "OK (6)" "$TMP/FLUID.log"
echo "FLUID_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/FLUID.log")"
exit 0
