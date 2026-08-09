#!/bin/bash
# Tier-2 for fourc::mixture#0 — the rule is right, the advice is unnecessary and
# the quantity is misnamed.
#
# Claimed: "the volume fractions must SUM TO 1 across all constituents at every
#          point.  Verify in pre-processing" — with the failure mode being
#          "nonphysical stress scaling (e.g. total > sum of constituents)".
#
# Observed: (a) the input is MASSFRAC, mass fractions, not volume fractions;
#           (b) you do not need to verify anything in pre-processing, because
#               MIX_Rule_Simple checks it itself at setup and refuses to run:
#
#     Mass fractions at element 0 sum to 0.8 instead of 1.0, which is unphysical.
#     src/mixture/src/4C_mixture_rule_simple.cpp
#
#           raised from SimpleMixtureRule::setup(), exit 1, in both directions
#           (under-sum and over-sum).  No stress is ever evaluated, so the
#           "total > sum of constituents" artefact cannot be produced.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream mixture_elast_hyper_dynamic.4C.yaml) || exit 3
grep -q "        constant: \[0.4, 0.6\]" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "      MASSFRAC:" "$BASE"                 || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/sum1.yaml"
sed 's/        constant: \[0.4, 0.6\]/        constant: [0.4, 0.4]/' "$BASE" > "$TMP/under.yaml"
sed 's/        constant: \[0.4, 0.6\]/        constant: [0.7, 0.6]/' "$BASE" > "$TMP/over.yaml"

probe SUM1  "$TMP/sum1.yaml"
probe UNDER "$TMP/under.yaml"
probe OVER  "$TMP/over.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SUM1.log"
grep -m1 -F "Mass fractions at element 0 sum to 0.8 instead of 1.0, which is unphysical." "$TMP/UNDER.log"
grep -m1 -oF "4C_mixture_rule_simple.cpp" "$TMP/UNDER.log"
echo "CHECKED_IN_SIMPLE_MIXTURE_RULE_SETUP=$(grep -c 'SimpleMixtureRule::setup' "$TMP/UNDER.log")"
# The over-sum direction is caught too.
grep -m1 -oF "instead of 1.0, which is unphysical." "$TMP/OVER.log"
# The input key is MASSFRAC — mass fractions, not volume fractions.
echo "UPSTREAM_USES_MASSFRAC=$(grep -c 'MASSFRAC' "$BASE")"
echo "UPSTREAM_USES_VOLFRAC=$(grep -c 'VOLFRAC' "$BASE")"
# No stress is ever evaluated, so no "total > sum of constituents" can be seen.
echo "UNDER_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/UNDER.log")"
exit 0
