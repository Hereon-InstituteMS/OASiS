#!/bin/bash
# Tier-2 for fourc::reduced_lung#0 — 4C prints the tree it built and then never
# asks whether that tree is a lung.
#
# Claim: "Airway tree TOPOLOGY must be physiologically reasonable — typically
#        16-23 generations (Weibel symmetric model) ... an unbalanced tree (e.g.
#        all generations branching into only 1 child) gives wrong total airway
#        resistance and tidal volume".
# Observed, on upstream reduced_lung_3_aw_2_tu (3 airways, 2 terminal units, one
# bifurcation): rewiring the same 3 airways into a serial chain with one terminal
# unit is accepted without comment. 4C's own setup banner is the whole story --
#     Bifurcations:         |  1     becomes     Bifurcations:         |  0
#     Terminal Units:       |  2                 Terminal Units:       |  1
# and the inlet flow halves, 8.3746 -> 4.1862, because half the parallel
# conductance is gone. There is no generation count, no acinus count, no alveolar
# surface, and no warning: the banner is a report, not a check.
#
# Numbers are read out of 4C's own ascii VTK cell data (q_in per element), which
# is why the deck is switched to OUTPUT_DATA_FORMAT: ascii.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream reduced_lung_3_aw_2_tu.4C.yaml) || exit 3
FIELDS=$(upstream reduced_lung_3_aw_2_tu_fields.json) || exit 3
cd "$TMP" || exit 3

# the second arm's tree shape.  "serial" is the pathology; "balanced" repeats the
# upstream tree and is what the mutation switches to.
TOPO=serial

python3 - "$BASE" "$FIELDS" "$TOPO" <<'PY'
import json, sys
t = open(sys.argv[1]).read()
assert '      num_nodes: 6\n      num_elements: 5' in t, "upstream topology size changed"
assert '    num_conditions: 3' in t, "upstream bc count changed"
t = t.replace('PROBLEM TYPE:', 'IO/RUNTIME VTK OUTPUT:\n  OUTPUT_DATA_FORMAT: ascii\nPROBLEM TYPE:')
open('bal.yaml', 'w').write(t)
open('bal_fields.json', 'w').write(open(sys.argv[2]).read())
if sys.argv[3] == 'serial':
    open('arm2.yaml', 'w').write(
        t.replace('      num_nodes: 6\n      num_elements: 5',
                  '      num_nodes: 5\n      num_elements: 4')
         .replace('    num_conditions: 3', '    num_conditions: 2'))
    json.dump({"node_coordinates": {"1": [0., 0., 0.], "2": [1., 0., 0.], "3": [2., 0., 0.],
                                    "4": [3., 0., 0.], "5": [4., 0., 0.]},
               "element_nodes": {"1": [1, 2], "2": [2, 3], "3": [3, 4], "4": [4, 5]},
               "element_type": {"1": "Airway", "2": "Airway", "3": "Airway", "4": "TerminalUnit"},
               "generation": {"1": 0, "2": 1, "3": 2, "4": -1},
               "radius": {"1": 1.5957691216057308, "2": 1.1283791670955126,
                          "3": 0.7978845608028654},
               "bc_node_id": {"1": 1, "2": 5},
               "bc_function_id": {"1": 1, "2": 2}}, open('arm2_fields.json', 'w'))
else:
    open('arm2.yaml', 'w').write(t)
    open('arm2_fields.json', 'w').write(open(sys.argv[2]).read())
PY

cp bal_fields.json reduced_lung_3_aw_2_tu_fields.json
probe BAL bal.yaml
cp arm2_fields.json reduced_lung_3_aw_2_tu_fields.json
probe UNBAL arm2.yaml

echo "--- balanced tree, as 4C reports it"
grep -A5 -F 'Instantiated objects' "$TMP/BAL.log" | grep -E 'Terminal Units|Bifurcations'
echo "--- second arm, as 4C reports it"
grep -A5 -F 'Instantiated objects' "$TMP/UNBAL.log" | grep -E 'Terminal Units|Bifurcations'
grep -m1 -F "processor 0 finished normally" "$TMP/BAL.log"
grep -m1 -F "processor 0 finished normally" "$TMP/UNBAL.log"

python3 - <<'PY'
import re
def qin(pref):
    t = open('o_%s-vtk-files/reduced_lung-00001-0.vtu' % pref).read()
    m = re.search(r'Name="q_in" format="ascii">(.*?)</DataArray>', t, re.S)
    return [float(x) for x in m.group(1).split()]
b, u = qin('BAL'), qin('UNBAL')
print("BAL_INLET_FLOW=%.6f" % b[0])
print("UNBAL_INLET_FLOW=%.6f" % u[0])
print("INLET_FLOW_RATIO=%.4f" % (b[0] / u[0]))
PY
# Nothing is checked about how lung-like the tree is.
echo "TREE_VALIDITY_WARNINGS=$(grep -ciE 'generation|weibel|alveol|unbalanc|not physiolog' "$TMP/UNBAL.log")"
exit 0
