#!/bin/bash
# Tier-2 for fourc::tsi#4 — TSI DYNAMIC owns the time loop; the per-field
# STRUCTURAL DYNAMIC / THERMAL DYNAMIC values for it are read and discarded,
# without a word.
#
# Upstream tsi_lincompression_1waydisp already demonstrates the TIMESTEP half by
# construction: STRUCTURAL DYNAMIC says TIMESTEP 0.01 / MAXTIME 1, TSI DYNAMIC
# says TIMESTEP 0.1 / MAXTIME 3, and the run marches at DT = 1.000e-01 for 30
# steps.  Three arms make the NUMSTEP half explicit:
#
#   BASE          untouched                       -> 30 steps, banner "/ 200"
#   STRUCT_NUMSTEP  NUMSTEP: 7 in STRUCTURAL DYNAMIC -> still 30 steps, still
#                                                    "/ 200", result tests pass
#   TSI_NUMSTEP     NUMSTEP: 7 in TSI DYNAMIC       -> 7 steps, banner "/   7",
#                                                    run stops early, exit 1
#
# 200 is TSI DYNAMIC's own default NUMSTEP, so the banner is showing whose value
# is in force.  No diagnostic mentions the ignored key.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_lincompression_1waydisp.4C.yaml) || exit 3
grep -q 'STRUCTURAL DYNAMIC:' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/base.yaml"

python3 - "$BASE" "$TMP/struct.yaml" "$TMP/tsi.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
s_anchor = 'STRUCTURAL DYNAMIC:\n  DYNAMICTYPE: "Statics"\n'
t_anchor = 'TSI DYNAMIC:\n  COUPALGO: "tsi_oneway"\n'
if s_anchor not in t or t_anchor not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(s_anchor, s_anchor + "  NUMSTEP: 7\n", 1))
open(sys.argv[3], "w").write(t.replace(t_anchor, t_anchor + "  NUMSTEP: 7\n", 1))
PY
[ -s "$TMP/struct.yaml" ] && [ -s "$TMP/tsi.yaml" ] || exit 3

probe BASE           "$TMP/base.yaml"
probe STRUCT_NUMSTEP "$TMP/struct.yaml"
probe TSI_NUMSTEP    "$TMP/tsi.yaml"

echo "STEPS_BASE=$(grep -c '^TIME: ' "$TMP/BASE.log")"
echo "STEPS_STRUCT_NUMSTEP_7=$(grep -c '^TIME: ' "$TMP/STRUCT_NUMSTEP.log")"
echo "STEPS_TSI_NUMSTEP_7=$(grep -c '^TIME: ' "$TMP/TSI_NUMSTEP.log")"
# The banner names the NUMSTEP actually in force.
grep -m1 -F "STEP =   30/ 200" "$TMP/STRUCT_NUMSTEP.log"
grep -m1 -F "STEP =    7/   7" "$TMP/TSI_NUMSTEP.log"
# The structural section's TIMESTEP (0.01) is not the one being used either.
echo "STRUCT_TIMESTEP_IN_DECK=$(grep -c 'TIMESTEP: 0.01' "$TMP/base.yaml")"
echo "MARCHED_AT_TSI_TIMESTEP=$(grep -c 'DT = 1.000e-01' "$TMP/BASE.log")"
# Ignoring the structural NUMSTEP costs nothing: the deck's own result tests pass.
grep -m1 -F "is CORRECT" "$TMP/STRUCT_NUMSTEP.log"
grep -m1 -F "processor 0 finished normally" "$TMP/STRUCT_NUMSTEP.log"
# ... and 4C never says the key was unused.
echo "IGNORED_KEY_DIAGNOSTIC=$(grep -ciE 'unused|ignored|not used|has no effect' "$TMP/STRUCT_NUMSTEP.log")"
exit 0
