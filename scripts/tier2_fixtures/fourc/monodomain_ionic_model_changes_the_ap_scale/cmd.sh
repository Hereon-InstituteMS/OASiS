#!/bin/bash
# Tier-2 for fourc::cardiac_monodomain#3 — MODEL in MAT_myocard changes the units
# the answer lives in, not just its shape.
#
# Claim: "Ionic cell model parameters (MODEL in MAT_myocard) must match the desired
#        physiology ... a FHN model on a problem expecting human ventricular
#        myocyte response gives wrong APD."
# Observed, on upstream scatra_myocard_FHN_material: swap MODEL "FHN" for "TNNP"
# (ten Tusscher-Noble-Noble-Panfilov) and change nothing else. The run converges
# and returns phi = -85.811 where the FHN deck's own result test expects 0.778036.
# The two models do not merely differ in action-potential duration: FHN's phi is a
# dimensionless 0..1 excitation variable and TNNP's is a transmembrane potential
# in mV whose rest is about -85, so the initial field, the stimulus amplitude and
# the result values all have to be re-derived. The upstream TNNP deck says so:
# its INITFUNCNO field is -85.423 and its Neumann VAL is 30, against 0.0 and 0.3
# for FHN. 4C accepts the mismatch silently -- MODEL is just an enum.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream scatra_myocard_FHN_material.4C.yaml) || exit 3
TNNP=$(upstream scatra_myocard_TNNP_material.4C.yaml) || exit 3
IFPACK=$(upstream xml/preconditioner/ifpack.xml) || exit 3
cd "$TMP" || exit 3
mkdir -p xml/preconditioner && cp "$IFPACK" xml/preconditioner/
grep -q '      MODEL: "FHN"' "$BASE"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '      MODEL: "TNNP"' "$TNNP" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
echo "UPSTREAM_FHN_REST_AND_STIMULUS=$(grep -c 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"' "$BASE")$(grep -c '    VAL: \[0.3\]' "$BASE")"
echo "UPSTREAM_TNNP_REST_AND_STIMULUS=$(grep -c 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "-85.423"' "$TNNP")$(grep -c '    VAL: \[30\]' "$TNNP")"

SWAPPED_MODEL=TNNP

sed 's/  RESULTSEVERY: 20/  RESULTSEVERY: 1000000/; s/  RESTARTEVERY: 20/  RESTARTEVERY: 1000000/' \
    "$BASE" > base.yaml
sed "s/      MODEL: \"FHN\"/      MODEL: \"$SWAPPED_MODEL\"/" base.yaml > swapped.yaml

probe BASE    base.yaml
probe SWAPPED swapped.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "SWAPPED_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SWAPPED.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "phi      at node   1" "$TMP/SWAPPED.log"
python3 - "$TMP" <<'PY'
import re, sys
for line in open(sys.argv[1] + '/SWAPPED.log'):
    m = re.search(r'phi      at node   1\s+is WRONG --> actresult=\s*(\S+)\s*,', line)
    if m:
        v = float(m.group(1))
        print("SWAPPED_MODEL_RESTS_NEAR_MINUS_85=%s" % ("yes" if -90 < v < -80 else "no"))
        print("SWAPPED_MODEL_STAYED_IN_FHN_RANGE=%s" % ("yes" if 0 <= v <= 1 else "no"))
        break
else:
    raise SystemExit("no phi result line in SWAPPED.log")
PY
echo "MODEL_MISMATCH_WARNINGS=$(grep -ciE 'ionic model|model mismatch|initial field.*model|APD' "$TMP/SWAPPED.log")"
exit 0
