#!/bin/bash
# Tier-2 for fourc::beam_interaction#1 — the beam-to-beam contact penalty has two
# failure modes and they look nothing alike.
#
# Claimed: "NOX condition-number printout exceeds ~1e14 (too high), or
#          post-processed beam-beam distance goes negative by more than 5% of beam
#          radius (too low)". No condition-number printout appears and no
#          post-processing is needed to see either.
# Observed, on upstream beam3eb_static_contact_penalty_linpen_..._twobeamstwisting
# with BEAMS_BTBLINEPENALTYPARAM = 4.68452798e4:
#   * /1e4  -> the solve CONVERGES and gives a silently wrong answer: node 5
#     dispx comes out 4.4e-6 where the deck expects -1.90e-2. Contact is simply
#     not enforced; only the result test notices.
#   * x1e8  -> NOX gives up outright: "The nonlinear solver did not converge!"
#     from solver_nonlin_nox/4C_solver_nonlin_nox_problem.cpp, and no result test
#     runs at all.
# Under-penalty is the dangerous one because it produces a plausible number.
. "$(dirname "$0")/../_lib/preamble.sh"

DECK=beam3eb_static_contact_penalty_linpen_limitdispperiter_twobeamstwisting
BASE=$(upstream "$DECK.4C.yaml") || exit 3
XML=$(upstream "$DECK.xml")      || exit 3
cd "$TMP" || exit 3
cp "$XML" .
cp "$BASE" base.yaml
grep -q "BEAMS_BTBLINEPENALTYPARAM: 46845.27980953355" base.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

sed 's/BEAMS_BTBLINEPENALTYPARAM: 46845.27980953355/BEAMS_BTBLINEPENALTYPARAM: 4.684527980953355/'      base.yaml > lowpen.yaml
sed 's/BEAMS_BTBLINEPENALTYPARAM: 46845.27980953355/BEAMS_BTBLINEPENALTYPARAM: 4.684527980953355e+12/'  base.yaml > highpen.yaml

probe BASE    base.yaml
probe LOWPEN  lowpen.yaml
probe HIGHPEN highpen.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "LOWPEN_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/LOWPEN.log")"
echo "LOWPEN_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/LOWPEN.log")"
echo "HIGHPEN_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/HIGHPEN.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/LOWPEN.log"
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/HIGHPEN.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/HIGHPEN.log"
# No condition-number printout exists in either arm.
echo "CONDITION_NUMBER_PRINTOUTS=$(grep -ci 'condition number' "$TMP/LOWPEN.log" "$TMP/HIGHPEN.log" | awk -F: '{s+=$2} END {print s+0}')"
exit 0
