#!/bin/bash
# Tier-2 for fourc::arterial_network#5 — a local aneurysm-like bulge is accepted
# without a validity check and does not give a "smoothly-varying" anything.
#
# Claimed: "applying 1D to a saccular aneurysm gives smoothly-varying pressure but
#          misses the stagnation region and wall stress concentration ... The 1D
#          result is correct for global hemodynamics (pulse propagation) only."
# Observed, on upstream one_d_3_artery_network shortened to 50 steps, by widening
# ONE of the four trunk segments:
#   * DIAM 24 -> 26.4 (a 10% bulge). The run completes and the answer is garbage:
#     flowrate at node 4 comes out -7.36e+05 where the same deck without the bulge
#     gives 2.33e+01, and the area at nodes 2 and 3 splits 326 / 509 around the
#     step. Not a smooth field, and not merely a missing stagnation region.
#   * DIAM 24 -> 48 (a 2x bulge). 4C dies with a bare SIGFPE inside
#     evaluate_wf_and_wb: shell status 136, no "PROC 0 ERROR" banner at all
#     (BULGE2X_ERROR_BANNERS=0), so there is nothing to grep for.
#   * neither arm prints one word about axisymmetry, flow separation or the
#     validity of the 1D reduction.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream one_d_3_artery_network.4C.yaml) || exit 3
cd "$TMP" || exit 3
grep -q '  NUMSTEP: 10000' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '"3 ART LINE2 3 4 MAT 1 GP 5 TYPE LinExp DIAM 24.0"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

BULGE_DIAM=26.4

sed 's/  NUMSTEP: 10000/  NUMSTEP: 50/' "$BASE" > base.yaml
sed "s|\"3 ART LINE2 3 4 MAT 1 GP 5 TYPE LinExp DIAM 24.0\"|\"3 ART LINE2 3 4 MAT 1 GP 5 TYPE LinExp DIAM $BULGE_DIAM\"|" base.yaml > bulge.yaml
sed 's|"3 ART LINE2 3 4 MAT 1 GP 5 TYPE LinExp DIAM 24.0"|"3 ART LINE2 3 4 MAT 1 GP 5 TYPE LinExp DIAM 48.0"|' base.yaml > bulge2x.yaml

probe BASE    base.yaml
probe BULGE   bulge.yaml
probe BULGE2X bulge2x.yaml

echo "BULGE_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/BULGE.log")"
grep -m1 -F "flowrate at node   4" "$TMP/BASE.log"
grep -m1 -F "flowrate at node   4" "$TMP/BULGE.log"
grep -m1 -F "area     at node   2" "$TMP/BULGE.log"
grep -m1 -F "area     at node   3" "$TMP/BULGE.log"
python3 - "$TMP" <<'PY'
import re, sys
def val(p, what):
    for line in open(p):
        m = re.search(what + r'\s+is WRONG --> actresult=\s*(\S+),', line)
        if m:
            return float(m.group(1))
    raise SystemExit("no %s line in %s" % (what, p))
t = sys.argv[1]
b = val(t + '/BASE.log', 'flowrate at node   4')
g = val(t + '/BULGE.log', 'flowrate at node   4')
print("BULGE_FLOW_BLOWUP_FACTOR=%.0f" % abs(g / b))
print("BULGE_FLOW_KEPT_ITS_SIGN=%s" % ("yes" if b * g > 0 else "no"))
PY
# the 2x bulge dies without any banner to grep for
echo "BULGE2X_ERROR_BANNERS=$(grep -c 'PROC 0 ERROR' "$TMP/BULGE2X.log")"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/BULGE2X.log"
grep -m1 -F "evaluate_wf_and_wb" "$TMP/BULGE2X.log"
# nothing about the 1D reduction's validity in either arm
echo "VALIDITY_WARNINGS=$(grep -ciE 'axisymmetr|flow separation|assumption|aneurysm|not valid' "$TMP/BULGE.log" "$TMP/BULGE2X.log" | awk -F: '{s+=$2} END {print s+0}')"
exit 0
