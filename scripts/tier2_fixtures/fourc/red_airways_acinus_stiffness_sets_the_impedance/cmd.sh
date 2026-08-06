#!/bin/bash
# Tier-2 for fourc::reduced_airways#2 — where the terminal impedance actually
# comes from, and what a compliance-free acinus really does.
#
# Claimed: "Missing acinar conditions give open-ended branches ... terminal-branch
#          pressure equals the upstream pressure at all times (no impedance), and
#          flow into the terminal segment grows unbounded; tidal volume diverges
#          with cycle count."
# Observed, on upstream red_airway_one_acinus_NeoHookean (one flow-driven
# RED_ACINUS, downstream pressure held at 0):
#   * the terminal impedance is the MAT_0D_MAXWELL_ACINUS_* stiffness/viscosity,
#     not a separate "acinar condition". Dividing all four by 1e6 collapses the
#     inlet pressure from 521.378 to 5.214e-4 -- i.e. to the downstream pressure,
#     which is the "no impedance" half of the claim, exactly and to six digits.
#   * the volume half is wrong. The acinar volume is 1023.6 in BOTH arms, to 12
#     digits, because it is fixed by the prescribed inflow: a hyper-compliant
#     acinus does not make the tidal volume diverge.
#   * 4C prints nothing at all about the acinus being too soft.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream red_airway_one_acinus_NeoHookean.4C.yaml) || exit 3
cd "$TMP" || exit 3

python3 - "$BASE" <<'PY'
import sys
t = open(sys.argv[1]).read()
# a volume probe: an impossible target makes 4C print the true acini_volume
t = t.rstrip('\n') + '''
  - RED_AIRWAY:
      DIS: "red_airway"
      ELEMENT: 1
      QUANTITY: "acini_volume"
      VALUE: -12345.0
      TOLERANCE: 1e-12
'''
open('base.yaml', 'w').write(t)
soft = t
for a, b in [("Stiffness1: 14647.5", "Stiffness1: 0.0146475"),
             ("Stiffness2: 5352.59", "Stiffness2: 0.00535259"),
             ("Viscosity1: 3285.38", "Viscosity1: 0.00328538"),
             ("Viscosity2: 188.023", "Viscosity2: 0.000188023")]:
    assert a in soft, "upstream acinus material changed: " + a
    soft = soft.replace(a, b)
open('soft.yaml', 'w').write(soft)
PY

probe BASE base.yaml
probe SOFT soft.yaml

# The inlet pressure collapses onto the prescribed downstream pressure.
grep -m1 -F "pressure at node   1" "$TMP/BASE.log"
grep -m1 -F "pressure at node   1" "$TMP/SOFT.log"
echo "SOFT_INLET_PRESSURE_IS_MILLIONTH=$(grep -c 'pressure at node   1.*actresult= 5.2137815275311[0-9]*e-04' "$TMP/SOFT.log")"
grep -m1 -F '|Pressure|_max:  5.214E-04' "$TMP/SOFT.log"
# but the acinar volume does not diverge: it is the same in both arms.
grep -m1 -F "acini_volume at element   1" "$TMP/BASE.log"
grep -m1 -F "acini_volume at element   1" "$TMP/SOFT.log"
python3 - "$TMP" <<'PY'
import re, sys
def vol(p):
    for line in open(p):
        m = re.search(r'acini_volume at element\s+1\s+is WRONG --> actresult=\s*(\S+),', line)
        if m:
            return float(m.group(1))
    raise SystemExit("no acini_volume probe line in " + p)
t = sys.argv[1]
b, s = vol(t + '/BASE.log'), vol(t + '/SOFT.log')
print("ACINUS_VOLUME_RELATIVE_CHANGE_UNDER_1E-12=%s" % ("yes" if abs(b - s) / abs(b) < 1e-12 else "no"))
print("VERDICT: HYPERCOMPLIANT_ACINUS_DIVERGES_TIDAL_VOLUME=%s"
      % ("no" if abs(b - s) / abs(b) < 1e-12 else "yes"))
PY
echo "SOFT_ACINUS_WARNINGS=$(grep -ciE '(warning|error|caution).*(acinus|acinar|compliance|impedance)' "$TMP/SOFT.log")"
exit 0
