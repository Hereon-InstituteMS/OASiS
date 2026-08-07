#!/bin/bash
# Tier-2 for fourc::beams#2 — TRIADS really is required on a BEAM3R element line,
# but NOT for the reason the entry gave.
#
# Claimed:  omitting TRIADS "gives zero initial rotation reference" and the first
#           load step then computes finite rotations from an undefined
#           configuration — i.e. the run proceeds and produces garbage.
# Observed: the run does not start. The element line fails to parse:
#
#     Required 'one_of' not found in input line
#     .../core/io/src/4C_io_input_spec_builders.cpp, line 111
#
#   because BEAM3R's spec is all_of({MAT, one_of({TRIADS, NODAL_ROTATION_VECTORS}),
#   USE_FAD, HERMITE_CENTERLINE}) — one of the two triad sources is mandatory.
#   Nothing is ever "silently zero", and no time step is ever taken.
#
# The diagnostic is also notably unhelpful: it names neither TRIADS nor BEAM3R.
# DIAGNOSTIC_NAMES_TRIADS=no keeps that pinned, because an agent grepping the
# error for "TRIADS" learns nothing about what it left out.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_static_test1.4C.yaml) || exit 3
grep -q "TRIADS" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/with_triads.yaml"

python3 - "$BASE" "$TMP/no_triads.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
out = re.sub(r' TRIADS [0-9. ]+ USE_FAD', ' USE_FAD', t)
assert out != t, "upstream deck no longer carries 'TRIADS ... USE_FAD' element lines"
open(sys.argv[2], "w").write(out)
PY

probe WITHTRIADS "$TMP/with_triads.yaml"
probe NOTRIADS   "$TMP/no_triads.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHTRIADS.log"
grep -m1 -F "Required 'one_of' not found in input line" "$TMP/NOTRIADS.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/NOTRIADS.log"
# It dies inside the element reader, before any discretisation exists...
grep -m1 -F "ElementReader::get_and_distribute_elements" "$TMP/NOTRIADS.log"
# ...so no time step is ever taken: the claimed "first load step" never happens.
echo "NOTRIADS_STEPS_TAKEN=$(grep -c 'Finalised step' "$TMP/NOTRIADS.log")"
# And the diagnostic names neither the missing keyword nor the element.
if grep -qE "TRIADS|BEAM3R" "$TMP/NOTRIADS.log"; then
  echo "DIAGNOSTIC_NAMES_TRIADS=yes"
else
  echo "DIAGNOSTIC_NAMES_TRIADS=no"
fi
exit 0
