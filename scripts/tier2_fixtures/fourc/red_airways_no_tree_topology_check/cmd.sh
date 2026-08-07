#!/bin/bash
# Tier-2 for fourc::reduced_airways#0 — there is no airway-tree topology check.
#
# Claimed: "4C aborts during topology check with `node X has degree 1 (dangling)`
#          or `cycle detected in airway tree`".
# Observed, on upstream red_airway_3airway_2acinus_awacinter (3 airways, 2 acini,
# one bifurcation): adding a sixth element that closes a loop 2-3-4-2, and adding
# a sixth element that hangs a free node 7 off node 4 with no boundary condition,
# are BOTH accepted. Each deck runs its full 5000-step time loop, reaches the
# result test, and prints not one word about topology, degree, dangling nodes or
# cycles. The only thing that notices is the deck's own RESULT DESCRIPTION: the
# cycle shifts node 2 pressure by 8e-2 and the dangling branch by 1.0e+1 (a third
# of the answer). A tree that is not a tree is a silent wrong answer, not an abort.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream red_airway_3airway_2acinus_awacinter.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" base.yaml
grep -q '"5 RED_ACINUS LINE2 4 6 MAT 2 TYPE Exponential' base.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '"NODE 6 COORD 17.00 -2.000 0.000"' base.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - <<'PY'
t = open('base.yaml').read()
anchor = '''  - "5 RED_ACINUS LINE2 4 6 MAT 2 TYPE Exponential AcinusVolume 1.0 AlveolarDuctVolume 1.0 E1_0 8.0 E1_LIN
    1.0 E1_EXP 0.022 TAU 7"
'''
assert anchor in t
def extra(n1, n2):
    return ('  - "6 RED_AIRWAY LINE2 %d %d MAT 1 ElemSolvingType NonLinear TYPE '
            'ConvectiveViscoElasticRLC Resistance\n    Poiseuille PowerOfVelocityProfile 2 '
            'WallElasticity 500.0 PoissonsRatio 0.4 ViscousTs 2.0 ViscousPhaseShift\n'
            '    0.13 WallThickness 0.1 Area 1.0 Generation 2"\n' % (n1, n2))
# a loop: 2 -> 3 -> 4 -> 2
open('cycle.yaml', 'w').write(t.replace(anchor, anchor + extra(4, 3)))
# a dangling branch: node 7 has degree 1 and carries no boundary condition
open('dangling.yaml', 'w').write(
    t.replace(anchor, anchor + extra(4, 7))
     .replace('  - "NODE 6 COORD 17.00 -2.000 0.000"',
              '  - "NODE 6 COORD 17.00 -2.000 0.000"\n  - "NODE 7 COORD 20.00 -6.000 0.000"'))
PY

probe BASE     base.yaml
probe CYCLE    cycle.yaml
probe DANGLING dangling.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "CYCLE_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/CYCLE.log")"
echo "DANGLING_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/DANGLING.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/CYCLE.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/DANGLING.log"
# Nothing is said about topology in either arm.
echo "TOPOLOGY_DIAGNOSTICS=$(grep -ciE 'dangling|cycle detected|degree 1|topolog' "$TMP/CYCLE.log" "$TMP/DANGLING.log" | awk -F: '{s+=$2} END {print s+0}')"
echo "CLAIMED_DANGLING_TEXT=$(grep -ci 'has degree 1' "$TMP/DANGLING.log")"
echo "CLAIMED_CYCLE_TEXT=$(grep -ci 'cycle detected in airway tree' "$TMP/CYCLE.log")"
exit 0
