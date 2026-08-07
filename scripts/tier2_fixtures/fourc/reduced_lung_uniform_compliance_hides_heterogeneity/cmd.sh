#!/bin/bash
# Tier-2 for fourc::reduced_lung#1 — the terminal-unit stiffness can be set per
# acinus, and leaving it uniform is a decision 4C will not question.
#
# Claim: "Alveolar compliance varies with DISEASE STATE ... using a uniform
#        compliance for a diseased-lung model misses the spatial heterogeneity
#        ... Vary compliance per acinus by region."
# Observed, on upstream reduced_lung_3_aw_2_tu (two terminal units, elements 4
# and 5): the knob is reduced_dimensional_lung.lung_tree.terminal_units.
# elasticity_model.linear.elasticity_e, and it takes `constant:` or `from_file:`.
# With `constant: 1.0` the two acini are interchangeable and split the flow
# 4.18805 / 4.18657. Swapping in a heterogeneous field with the SAME mean --
# 0.05 for element 4, 1.95 for element 5 -- gives 83.5285 / 2.1463, a 38.9 : 1
# split, and a total inlet flow 10x higher. Same mean stiffness, completely
# different ventilation. 4C prints no warning in either case.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream reduced_lung_3_aw_2_tu.4C.yaml) || exit 3
FIELDS=$(upstream reduced_lung_3_aw_2_tu_fields.json) || exit 3
cd "$TMP" || exit 3

# per-acinus stiffnesses for the second arm.  Mean is 1.0, i.e. the uniform value.
E_ACINUS_4=0.05
E_ACINUS_5=1.95

python3 - "$BASE" "$FIELDS" "$E_ACINUS_4" "$E_ACINUS_5" <<'PY'
import json, sys
t = open(sys.argv[1]).read()
uni = '          elasticity_e:\n            constant: 1.0'
assert uni in t, "upstream terminal-unit elasticity block changed"
t = t.replace('PROBLEM TYPE:', 'IO/RUNTIME VTK OUTPUT:\n  OUTPUT_DATA_FORMAT: ascii\nPROBLEM TYPE:')
open('uniform.yaml', 'w').write(t)
open('hetero.yaml', 'w').write(t.replace(
    uni, '          elasticity_e:\n            from_file: "reduced_lung_3_aw_2_tu_fields.json"'))
f = json.load(open(sys.argv[2]))
open('uniform_fields.json', 'w').write(json.dumps(f))
f['elasticity_e'] = {"4": float(sys.argv[3]), "5": float(sys.argv[4])}
open('hetero_fields.json', 'w').write(json.dumps(f))
PY

cp uniform_fields.json reduced_lung_3_aw_2_tu_fields.json
probe UNIFORM uniform.yaml
cp hetero_fields.json reduced_lung_3_aw_2_tu_fields.json
probe HETERO hetero.yaml

grep -m1 -F "processor 0 finished normally" "$TMP/UNIFORM.log"
grep -m1 -F "processor 0 finished normally" "$TMP/HETERO.log"
python3 - <<'PY'
import re
def qin(pref):
    t = open('o_%s-vtk-files/reduced_lung-00001-0.vtu' % pref).read()
    m = re.search(r'Name="q_in" format="ascii">(.*?)</DataArray>', t, re.S)
    return [float(x) for x in m.group(1).split()]
u, h = qin('UNIFORM'), qin('HETERO')
print("UNIFORM_ACINAR_FLOWS=%.5f %.5f" % (u[3], u[4]))
print("HETERO_ACINAR_FLOWS=%.4f %.4f" % (h[3], h[4]))
print("UNIFORM_ACINAR_FLOW_SPLIT=%.4f" % (u[3] / u[4]))
print("HETERO_ACINAR_FLOW_SPLIT=%.4f" % (h[3] / h[4]))
print("HETERO_TOTAL_FLOW_FACTOR=%.3f" % (h[0] / u[0]))
PY
echo "HETEROGENEITY_WARNINGS=$(grep -ciE 'compliance|heterogene|elasticity_e' "$TMP/UNIFORM.log")"
exit 0
