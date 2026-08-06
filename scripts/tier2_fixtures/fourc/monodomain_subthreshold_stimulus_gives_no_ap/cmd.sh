#!/bin/bash
# Tier-2 for fourc::cardiac_monodomain#5 — a subthreshold stimulus produces a
# perfectly converged nothing.
#
# Claim: "The stimulus must be SUPRATHRESHOLD to trigger an action potential.
#        Signal: too-small stimulus produces NO AP at all (subthreshold)."
# Observed, on upstream scatra_myocard_FHN_material, whose DESIGN VOL NEUMANN
# CONDITIONS carries VAL: [0.3]: dividing that by 100 leaves the run converging
# normally through all 9000 steps and ending at phi EXACTLY 0.0 -- the FitzHugh-
# Nagumo resting state -- against 0.778036 for the calibrated stimulus. There is
# no threshold check, no warning, and no hint in the output that the tissue was
# never excited; only the deck's own result test notices.
# Halving the stimulus instead (0.3 -> 0.15) still fires, which is what makes the
# failure a cliff rather than a gradient.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream scatra_myocard_FHN_material.4C.yaml) || exit 3
IFPACK=$(upstream xml/preconditioner/ifpack.xml) || exit 3
cd "$TMP" || exit 3
mkdir -p xml/preconditioner && cp "$IFPACK" xml/preconditioner/
grep -q '    VAL: \[0.3\]' "$BASE"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  RESULTSEVERY: 20' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

SUBTHRESHOLD_VAL=0.003

sed 's/  RESULTSEVERY: 20/  RESULTSEVERY: 1000000/; s/  RESTARTEVERY: 20/  RESTARTEVERY: 1000000/' \
    "$BASE" > base.yaml
sed "s/    VAL: \[0.3\]/    VAL: [$SUBTHRESHOLD_VAL]/" base.yaml > sub.yaml
sed 's/    VAL: \[0.3\]/    VAL: [0.15]/'              base.yaml > half.yaml

probe BASE base.yaml
probe SUB  sub.yaml
probe HALF half.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "SUB_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SUB.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "phi      at node   1" "$TMP/SUB.log"
grep -m1 -F "phi      at node   1" "$TMP/HALF.log"
if grep -q 'phi      at node   1.*actresult= 0.00000000000000000e+00' "$TMP/SUB.log"; then
  echo "VERDICT: SUBTHRESHOLD_STIMULUS_FIRES_AN_AP=no"
else
  echo "VERDICT: SUBTHRESHOLD_STIMULUS_FIRES_AN_AP=yes"
fi
python3 - "$TMP" <<'PY'
import re, sys
def val(p):
    for line in open(p):
        m = re.search(r'phi      at node   1\s+is WRONG --> actresult=\s*(\S+)\s*,', line)
        if m:
            return float(m.group(1))
    return None
h = val(sys.argv[1] + '/HALF.log')
print("HALF_STIMULUS_STILL_FIRES=%s" % ("yes" if (h is None or h > 0.5) else "no"))
PY
echo "SUB_THRESHOLD_WARNINGS=$(grep -ciE 'subthreshold|action potential|no activation|not excited' "$TMP/SUB.log")"
exit 0
