#!/bin/bash
# Tier-2 for fourc::arterial_network#2 — there is no Windkessel outlet in 4C's
# ArterialNetwork. The terminal knob is a reflection coefficient.
#
# Claimed: "Windkessel parameters (R, C) at terminal outlets strongly affect
#          REFLECTED waves ... Tune R from physiological impedance
#          Z_terminal = rho*c/A_terminal."
# Observed, on upstream one_d_3_artery_network shortened to 50 steps:
#   * writing a Windkessel outlet section is rejected outright:
#     "Section 'DESIGN NODE 1D ARTERY WINDKESSEL CONDITIONS' is not a valid
#     section name." There is no R, no C and no R_d to tune anywhere in the
#     ArterialNetwork input. (An ArtWkCond name survives in the element evaluator
#     but no input section ever creates it.)
#   * the terminal condition 4C really offers is
#     DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS with one number, the reflection
#     coefficient. It is the knob the claim is reaching for: raising it from the
#     upstream 0 to 0.9 at both outlets moves the distal flow 0.92858 -> 1.23276,
#     a 33% change, with the geometry and material untouched.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream one_d_3_artery_network.4C.yaml) || exit 3
cd "$TMP" || exit 3
grep -q '  NUMSTEP: 10000' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
sed 's/  NUMSTEP: 10000/  NUMSTEP: 50/' "$BASE" > base.yaml

python3 - <<'PY'
t = open('base.yaml').read()
refl = '''DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS:
  - E: 4
    VAL: [0]
    curve: [null]
  - E: 6
    VAL: [0]
    curve: [null]'''
assert refl in t, "upstream reflective conditions changed"
WK_SECTION = 'DESIGN NODE 1D ARTERY WINDKESSEL CONDITIONS:\n  - E: 4\n    R_p: 1.0\n    C: 1.0\n    R_d: 1.0\n'
open('windkessel.yaml', 'w').write(t.replace(
    'DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS:',
    WK_SECTION + 'DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS:', 1))
open('reflect.yaml', 'w').write(t.replace(refl, refl.replace('VAL: [0]', 'VAL: [0.9]')))
PY

probe BASE       base.yaml
probe WINDKESSEL windkessel.yaml
probe REFLECT    reflect.yaml

grep -m1 -F "Section 'DESIGN NODE 1D ARTERY WINDKESSEL CONDITIONS' is not a valid section name." "$TMP/WINDKESSEL.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/WINDKESSEL.log"
echo "WINDKESSEL_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/WINDKESSEL.log")"
# the terminal condition that does exist, and what it does
grep -m1 -F "flowrate at node  10" "$TMP/BASE.log"
grep -m1 -F "flowrate at node  10" "$TMP/REFLECT.log"
python3 - "$TMP" <<'PY'
import re, sys
def val(p):
    for line in open(p):
        m = re.search(r'flowrate at node\s+10\s+is WRONG --> actresult=\s*(\S+),', line)
        if m:
            return float(m.group(1))
    raise SystemExit("no distal flowrate line in " + p)
b, r = val(sys.argv[1] + '/BASE.log'), val(sys.argv[1] + '/REFLECT.log')
print("REFLECTION_COEFF_FLOW_CHANGE_PERCENT=%.1f" % (100.0 * (r - b) / b))
PY
exit 0
