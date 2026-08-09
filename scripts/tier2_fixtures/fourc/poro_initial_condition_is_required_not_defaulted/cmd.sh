#!/bin/bash
# Tier-2 for fourc::porous_media#4 — and a FALSIFICATION of how it was worded.
#
# Claimed: "Without an initial condition for pressure, the solver starts from
#           zero, which may be unphysical" — i.e. a silent, wrong-answer trap.
#
# Observed: you cannot omit it.  `initial_condition` is a REQUIRED group inside
# `porofluid_dynamic` (its `type` parameter has no default), so deleting it is a
# parse abort:
#
#     Could not match this input ... [X] Expected group 'initial_condition'
#
# from core/io/src/4C_io_input_spec_builders.cpp.  The run never starts, so
# there is no silent zero start to be had.  And if you DO want to start from
# zero, that is an explicit, supported value: `type: "zero"` parses and runs.
#
# Three arms: shipped by_function, the same deck with initial_condition deleted,
# and the same deck with type: "zero".
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream porofluid_pressure_based_2D_quad4.4C.yaml) || exit 3
ln -s "$(dirname "$BASE")/xml" "$TMP/xml"
grep -q 'type: "by_function"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/byfunct.yaml"
sed 's/    type: "by_function"/    type: "zero"/' "$BASE" > "$TMP/zerofield.yaml"
python3 - "$BASE" "$TMP/omitted.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = '  initial_condition:\n    type: "by_function"\n    function_id: 1\n'
if blk not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(blk, "", 1))
PY
[ -f "$TMP/omitted.yaml" ] || exit 3

probe BYFUNCT   "$TMP/byfunct.yaml"
probe OMITTED   "$TMP/omitted.yaml"
probe ZEROFIELD "$TMP/zerofield.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BYFUNCT.log"
grep -m1 -F "Could not match this input" "$TMP/OMITTED.log"
grep -m1 -F "[X] Expected group 'initial_condition'" "$TMP/OMITTED.log"
grep -m1 -oF "4C_io_input_spec_builders.cpp" "$TMP/OMITTED.log"
# The claimed silent zero start cannot happen: nothing runs.
echo "OMITTED_TIME_STEPS_STARTED=$(grep -c 'PORO MULTIPHASE FLUID SOLVER' "$TMP/OMITTED.log")"
# Starting from zero is an explicit, legal choice — and on this deck it does not
# even change the answer, because the Dirichlet drive washes the transient out.
echo "ZEROFIELD_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ZEROFIELD.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/ZEROFIELD.log"
exit 0
