#!/bin/bash
# Tier-2 for fourc::cardiovascular0d#0 — where R and C actually live, and how hard
# they bite.
#
# Claim: "Windkessel parameters (R, C) must MATCH the vascular impedance ...
#        arbitrary R, C give non-physiological pressure waveforms."
# Observed, on upstream cardiovascular0d_4elementwindkessel_structure_direct_
# genalpha, whose three cavities each carry a
# DESIGN SURF CARDIOVASCULAR 0D 4-ELEMENT WINDKESSEL CONDITIONS entry with
# C / R_p / Z_c / L / p_ref / p_0:
#   * R_p 5 -> 500 on cavity 0 alone moves the wall displacement it loads,
#     node 5 dispx -0.235921 -> -0.221994, and breaks exactly one of the deck's
#     three result tests. The other two cavities are untouched, so the effect is
#     local to the condition you mis-tuned.
#   * C 1.5 -> 0.015 is far more violent: node 5 dispx collapses to -0.017226,
#     a factor 13.7, because a stiff Windkessel refuses to accept volume.
#   * neither arm prints a word about the parameters being unphysiological.
#
# The deck's STRUCT NOX/Status Test names a relative XML file, which 4C resolves
# against the WORKING DIRECTORY: without it the run dies inside Teuchos
# FileInputStream with shell status 134 and no "PROC 0 ERROR" banner at all.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream cardiovascular0d_4elementwindkessel_structure_direct_genalpha.4C.yaml) || exit 3
NOXXML=$(upstream cardiovascular0d_new_struc.xml) || exit 3
cd "$TMP" || exit 3
cp "$NOXXML" .
cp "$BASE" base.yaml
grep -q 'XML File: "cardiovascular0d_new_struc.xml"' base.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
python3 - <<'PY'
t = open('base.yaml').read()
assert '    C: 1.5\n    R_p: 5\n' in t, "upstream windkessel cavity 0 changed"
PY

BIG_R_P=500

python3 - "$BIG_R_P" <<'PY'
import sys
t = open('base.yaml').read()
open('bigr.yaml', 'w').write(t.replace('    C: 1.5\n    R_p: 5\n',
                                       '    C: 1.5\n    R_p: %s\n' % sys.argv[1]))
open('smallc.yaml', 'w').write(t.replace('    C: 1.5\n    R_p: 5\n',
                                         '    C: 0.015\n    R_p: 5\n'))
PY

probe BASE   base.yaml
probe BIGR   bigr.yaml
probe SMALLC smallc.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "BIGR_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/BIGR.log")"
echo "SMALLC_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/SMALLC.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "dispx    at node   5" "$TMP/BIGR.log"
grep -m1 -F "dispx    at node   5" "$TMP/SMALLC.log"
python3 - "$TMP" <<'PY'
import re, sys
def val(p):
    for line in open(p):
        m = re.search(r'dispx    at node   5\s+is WRONG --> actresult=\s*(\S+),', line)
        if m:
            return float(m.group(1))
    raise SystemExit("no node-5 dispx failure in " + p)
t = sys.argv[1]
print("BIGR_WALL_MOTION_CHANGE_PERCENT=%.1f" % (100.0 * (val(t + '/BIGR.log') / -0.23592109321050103 - 1)))
print("SMALLC_WALL_MOTION_FACTOR=%.1f" % (-0.23592109321050103 / val(t + '/SMALLC.log')))
PY
echo "WINDKESSEL_PARAMETER_WARNINGS=$(grep -ciE 'unphysiolog|implausible|out of range|impedance' "$TMP/SMALLC.log")"
exit 0
