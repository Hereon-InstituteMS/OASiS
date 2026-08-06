#!/bin/bash
# Tier-2 for fourc::mixture#2 — right idea, invented key, wrong rule.
#
# Claimed: "Add the growth tensor F_g(t) driven by a concentration or stress
#          signal, with RHO_GROWTH FUNCT to specify time-varying mass density."
#
# Observed: RHO_GROWTH is not a 4C key, and growth is not something you bolt on
# to the steady rule.  It belongs to a DIFFERENT mixture rule:
#
#   GROWTH     upstream growth deck (MIX_GrowthRemodelMixtureRule +
#              MIX_GrowthStrategy_Stiffness + a full-constrained-mixture fibre
#              with GROWTH_CONSTANT and DECAY_TIME) -> runs, all tests pass
#   NOGROWTH   the same deck with GROWTH_CONSTANT zeroed -> results move, so
#              that constant IS the growth driver
#   RHOGROWTH  add RHO_GROWTH to the growth rule -> parse abort, no such key
#   SIMPLEGROW add GROWTH_STRATEGY to the STEADY rule (MIX_Rule_Simple, from the
#              non-growth deck) -> parse abort: the simple rule has no growth
#              slot at all, so "a steady mixture solve cannot capture growth" is
#              enforced by the input spec, not by a silent pure-elastic answer.
. "$(dirname "$0")/../_lib/preamble.sh"

GROW=$(upstream mixture_growth_full_constrained_mixture_fiber_non_adaptive.4C.yaml) || exit 3
SIMPLE=$(upstream mixture_elast_hyper_dynamic.4C.yaml) || exit 3
grep -q "      GROWTH_CONSTANT: 0.01" "$GROW"   || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "MIX_GrowthRemodelMixtureRule" "$GROW"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "    MIX_Rule_Simple:" "$SIMPLE"        || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$GROW" "$TMP/growth.yaml"
sed 's/      GROWTH_CONSTANT: 0.01/      GROWTH_CONSTANT: 0.0/' "$GROW" > "$TMP/nogrowth.yaml"
python3 - "$GROW" "$TMP/rhogrowth.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "      DENS: 1\n      MASSFRAC: [0.1, 0.9]\n"
if anchor not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(anchor, anchor + "      RHO_GROWTH: 1\n", 1))
PY
[ -f "$TMP/rhogrowth.yaml" ] || exit 3
python3 - "$SIMPLE" "$TMP/simplegrow.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
anchor = "    MIX_Rule_Simple:\n      DENS: 0.1\n"
if anchor not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(
    t.replace(anchor, "    MIX_Rule_Simple:\n      GROWTH_STRATEGY: 100\n      DENS: 0.1\n", 1))
PY
[ -f "$TMP/simplegrow.yaml" ] || exit 3

probe GROWTH     "$TMP/growth.yaml"
probe NOGROWTH   "$TMP/nogrowth.yaml"
probe RHOGROWTH  "$TMP/rhogrowth.yaml"
probe SIMPLEGROW "$TMP/simplegrow.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/GROWTH.log"
# GROWTH_CONSTANT is the driver: zero it and the answer changes.
echo "NOGROWTH_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOGROWTH.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NOGROWTH.log"
# RHO_GROWTH does not exist.
grep -m1 -F "Could not match this input" "$TMP/RHOGROWTH.log"
grep -m1 -oF "4C_global_data_read.cpp" "$TMP/RHOGROWTH.log"
# And the steady rule has no growth slot to attach one to.
echo "SIMPLE_RULE_ACCEPTS_GROWTH_STRATEGY=$(grep -c 'processor 0 finished normally' "$TMP/SIMPLEGROW.log")"
exit 0
