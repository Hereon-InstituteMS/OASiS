#!/bin/bash
# Tier-2 for fourc::arterial_network#1 — there is no AREA0 input in 4C, and the
# error from getting the reference area wrong is quartic, not a factor 2.
#
# Claimed: "AREA0 (reference cross-section area) must MATCH the actual vessel
#          geometry ... AREA0 set to half the physiological value gives pressure
#          off by ~2x".
# Observed, on upstream one_d_3_artery_network shortened to 50 steps:
#   * MAT_CNST_ART has no AREA0 parameter. Adding one fails to match section
#     'MATERIALS'. The reference area is A0 = pi*DIAM^2/4, taken from the DIAM
#     token on the ART element line.
#   * halving DIAM 24 -> 12 therefore quarters A0: the area result at node 1 goes
#     452.400 -> 113.105, which is pi*12^2/4 = 113.097 to five digits. Not 2x.
#   * the flow error is bigger still and in the other direction: flowrate at node
#     4 goes 23.266 -> 79.184, a factor 3.4 up, because the wave speed and the
#     junction impedances all move with A0 at once.
# So "off by ~2x" understates it, and the parameter it names does not exist.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream one_d_3_artery_network.4C.yaml) || exit 3
cd "$TMP" || exit 3
grep -q '  NUMSTEP: 10000' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '1 ART LINE2 1 2 MAT 1 GP 5 TYPE LinExp DIAM 24.0' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '      VISCOSITY: 0.04' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

sed 's/  NUMSTEP: 10000/  NUMSTEP: 50/' "$BASE" > base.yaml
sed 's/DIAM 24.0/DIAM 12.0/' base.yaml > halfdiam.yaml
python3 - <<'PY'
t = open('base.yaml').read()
old = '      VISCOSITY: 0.04\n      DENS: 0.001\n      YOUNG: 400000'
assert old in t
open('area0.yaml', 'w').write(
    t.replace(old, '      VISCOSITY: 0.04\n      DENS: 0.001\n      AREA0: 452.39\n'
                   '      YOUNG: 400000', 1))
PY

probe BASE     base.yaml
probe HALFDIAM halfdiam.yaml
probe AREA0    area0.yaml

# There is no AREA0 to set.
grep -m1 -F "Failed to match specification in section 'MATERIALS'." "$TMP/AREA0.log"
echo "AREA0_IS_A_MAT_CNST_ART_PARAMETER=$(grep -c 'Matched parameter .AREA0' "$TMP/AREA0.log")"
# The reference area is pi*DIAM^2/4 and halving DIAM quarters it.
grep -m1 -F "area     at node   1" "$TMP/BASE.log"
grep -m1 -F "area     at node   1" "$TMP/HALFDIAM.log"
grep -m1 -F "flowrate at node   4" "$TMP/BASE.log"
grep -m1 -F "flowrate at node   4" "$TMP/HALFDIAM.log"
python3 - "$TMP" <<'PY'
import re, sys
def val(p, what):
    for line in open(p):
        m = re.search(what + r'\s+is (?:WRONG --> )?actresult=\s*(\S+),', line)
        if m:
            return float(m.group(1))
    raise SystemExit("no %s line in %s" % (what, p))
t = sys.argv[1]
b = val(t + '/BASE.log', 'area     at node   1')
h = val(t + '/HALFDIAM.log', 'area     at node   1')
print("REFERENCE_AREA_RATIO=%.4f" % (b / h))
print("HALF_DIAM_PRESSURE_ERROR_IS_TWOFOLD=%s" % ("yes" if abs(b / h - 2.0) < 0.1 else "no"))
PY
exit 0
