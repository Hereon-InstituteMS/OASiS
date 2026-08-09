#!/bin/bash
# Tier-2 for fourc::reduced_lung#2 — a unit error at the 1D-airway / 0D-acinus
# interface is a clean multiplicative error and 4C never notices it.
#
# Claim: "unit mismatch at the interface (e.g. acinus expects Q in L/min but
#        airway delivers L/s) gives factor-60 error in tidal volume".
# Observed, on upstream reduced_lung_3_aw_2_tu: the linear terminal unit couples
# to the airway tree through elasticity_e, and the coupled system is linear in it.
# Feeding the acinus a stiffness that is 60x too large -- exactly the size of a
# seconds/minutes mix-up -- divides every flow in the tree by 59.978, i.e. the
# error passes straight through the interface with no distortion and no
# diagnostic. Inlet flow 8.374620 -> 0.139626, both runs exit 0.
# There is nothing at the interface that checks units; the answer is simply 60x
# wrong and looks perfectly well-behaved.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream reduced_lung_3_aw_2_tu.4C.yaml) || exit 3
FIELDS=$(upstream reduced_lung_3_aw_2_tu_fields.json) || exit 3
cd "$TMP" || exit 3
cp "$FIELDS" reduced_lung_3_aw_2_tu_fields.json

# the size of the unit slip fed to the 0D acinus
UNIT_SLIP=60.0

python3 - "$BASE" "$UNIT_SLIP" <<'PY'
import sys
t = open(sys.argv[1]).read()
uni = '          elasticity_e:\n            constant: 1.0'
assert uni in t, "upstream terminal-unit elasticity block changed"
t = t.replace('PROBLEM TYPE:', 'IO/RUNTIME VTK OUTPUT:\n  OUTPUT_DATA_FORMAT: ascii\nPROBLEM TYPE:')
open('consistent.yaml', 'w').write(t)
open('slipped.yaml', 'w').write(
    t.replace(uni, '          elasticity_e:\n            constant: %s' % sys.argv[2]))
PY

probe CONSISTENT consistent.yaml
probe SLIPPED    slipped.yaml

grep -m1 -F "processor 0 finished normally" "$TMP/CONSISTENT.log"
grep -m1 -F "processor 0 finished normally" "$TMP/SLIPPED.log"
python3 - <<'PY'
import re
def qin(pref):
    t = open('o_%s-vtk-files/reduced_lung-00001-0.vtu' % pref).read()
    m = re.search(r'Name="q_in" format="ascii">(.*?)</DataArray>', t, re.S)
    return [float(x) for x in m.group(1).split()]
c, s = qin('CONSISTENT'), qin('SLIPPED')
print("CONSISTENT_INLET_FLOW=%.6f" % c[0])
print("SLIPPED_INLET_FLOW=%.6f" % s[0])
print("FLOW_ERROR_FACTOR=%.3f" % (c[0] / s[0]))
ratios = [a / b for a, b in zip(c, s)]
print("ERROR_FACTOR_IS_UNIFORM_ACROSS_ELEMENTS=%s"
      % ("yes" if (max(ratios) - min(ratios)) / min(ratios) < 1e-3 else "no"))
PY
echo "UNIT_DIAGNOSTICS=$(grep -ciE '(wrong|inconsistent|mismatch).*(unit|dimension)' "$TMP/SLIPPED.log")"
exit 0
