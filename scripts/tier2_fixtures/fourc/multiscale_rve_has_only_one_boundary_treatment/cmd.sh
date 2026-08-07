#!/bin/bash
# Tier-2 for fourc::multiscale#3 — the comparison the entry recommends cannot be
# made inside 4C.
#
# Claimed: "homogenised tangent C* differs from a reference simulation with
#          periodic BCs by ~10-20%... (Dirichlet RVE over-constrains)".
# Observed: 4C's multiscale module offers exactly ONE RVE boundary treatment.
#          The micro input takes a MICROSCALE CONDITIONS section, which 4C
#          registers as the condition named MicroBoundary in
#          global_legacy_module/4C_global_legacy_module_validconditions.cpp, and
#          each entry carries a node-set id and nothing else. Adding any
#          boundary-type switch to it is rejected by
#          core/fem/src/condition/4C_fem_condition_definition.cpp. There is no
#          periodic option: the word does not occur anywhere in src/stru_multi.
#
# So an agent cannot run the periodic reference the entry tells it to compare
# against. The advice is sound as theory and unfollowable as instruction, which
# is what the fixture pins.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sohex8_multiscale_macro.4C.yaml) || exit 3
MICRO=$(upstream sohex8_multiscale_micro.mat.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" macro.yaml
cp "$MICRO" micro.yaml
sed -i 's|MICROFILE: "sohex8_multiscale_micro.mat.4C.yaml"|MICROFILE: "micro.yaml"|g' macro.yaml
grep -q "^MICROSCALE CONDITIONS:" micro.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - micro.yaml micro_periodic.yaml <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "MICROSCALE CONDITIONS:\n  - E: 1"
assert anchor in t, "upstream micro deck no longer has MICROSCALE CONDITIONS"
open(sys.argv[2], "w").write(
    t.replace(anchor, anchor + "\n    BCTYPE: \"periodic\"", 1))
PY
sed 's|MICROFILE: "micro.yaml"|MICROFILE: "micro_periodic.yaml"|g' macro.yaml > macro_per.yaml

probe GOOD     macro.yaml
probe PERIODIC macro_per.yaml

echo "GOOD_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/GOOD.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/GOOD.log"
grep -m1 -F "Could not match this input" "$TMP/PERIODIC.log"
grep -m1 -F "The following data remains unused" "$TMP/PERIODIC.log"
grep -m1 -F "4C_fem_condition_definition.cpp" "$TMP/PERIODIC.log"
# There is exactly one micro boundary section in the whole schema, and it has no
# type switch to set.
"$BIN" --parameters 2>/dev/null > params.json
echo "MICROSCALE_CONDITION_SECTIONS=$(grep -c 'MICROSCALE CONDITIONS' params.json)"
echo "PERIODIC_RVE_KEYS_IN_SCHEMA=$(grep -ci 'periodic_rve\|RVE_PERIODIC\|MICRO_BC' params.json)"
exit 0
