#!/bin/bash
# Tier-2 for fourc::ssti#2 — a FALSIFICATION.
#
# Claimed: "All three DYNAMIC sections (STRUCTURAL, SCALAR TRANSPORT, THERMAL)
#          must be consistently configured with MATCHING time step sizes.
#          Different TIMESTEP values produce sub-stepping which accumulates
#          O(dt) splitting error per outer iteration; result diverges from a
#          uniform-dt monolithic reference by 5-15%."
#
# Observed: there is nothing to match.  SSTI takes ONE time step, from SSTI
# CONTROL, and hands it to every field.  Put deliberately inconsistent TIMESTEP
# and NUMSTEP values into STRUCTURAL DYNAMIC and SCALAR TRANSPORT DYNAMIC and
# the run is bit-for-bit the reference: exit 0, every result test passes.  They
# are silently overridden, not sub-stepped.
#
#   MISMATCHED  STRUCTURAL TIMESTEP 0.7, SCALAR TRANSPORT TIMESTEP 0.9,
#               SSTI CONTROL TIMESTEP untouched -> identical result
#   SSTIDT      SSTI CONTROL TIMESTEP halved    -> result moves
#
# The second arm is the control: it proves the deck IS sensitive to its time
# step, so the first arm's zero change is an override and not a dead deck.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssti_mono_3D_3hex8_elch_s2i_butlervolmerthermo_growthlaw.4C.yaml) || exit 3
grep -q "^  TIMESTEP: 0.1$" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/ref.yaml"
python3 - "$BASE" "$TMP/mismatched.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
struct = 'STRUCTURAL DYNAMIC:\n  LINEAR_SOLVER: 1\n'
scatra = 'SCALAR TRANSPORT DYNAMIC:\n  SOLVERTYPE: "nonlinear"\n'
if struct not in t or scatra not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
t = t.replace(struct,
              'STRUCTURAL DYNAMIC:\n  TIMESTEP: 0.7\n  NUMSTEP: 3\n  LINEAR_SOLVER: 1\n', 1)
t = t.replace(scatra,
              'SCALAR TRANSPORT DYNAMIC:\n  TIMESTEP: 0.9\n  NUMSTEP: 2\n  SOLVERTYPE: "nonlinear"\n', 1)
open(sys.argv[2], "w").write(t)
PY
[ -f "$TMP/mismatched.yaml" ] || exit 3
sed 's/^  TIMESTEP: 0.1$/  TIMESTEP: 0.05/' "$BASE" > "$TMP/sstidt.yaml"

probe REF        "$TMP/ref.yaml"
probe MISMATCHED "$TMP/mismatched.yaml"
probe SSTIDT     "$TMP/sstidt.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/REF.log"
# Per-field TIMESTEP/NUMSTEP: accepted, and completely without effect.
grep -m1 -F "processor 0 finished normally" "$TMP/MISMATCHED.log"
echo "MISMATCHED_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/MISMATCHED.log")"
echo "MISMATCHED_WARNINGS=$(grep -ciE 'timestep.*(ignor|overrid|inconsist|mismatch)' "$TMP/MISMATCHED.log")"
# Control: the deck really does depend on its time step.
echo "SSTIDT_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SSTIDT.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/SSTIDT.log"
if [ "$(grep -c 'is WRONG --> actresult=' "$TMP/MISMATCHED.log")" = "0" ]; then
  echo "VERDICT: PER_FIELD_TIMESTEP_CHANGES_SSTI_RESULT=no"
else
  echo "VERDICT: PER_FIELD_TIMESTEP_CHANGES_SSTI_RESULT=yes"
fi
exit 0
