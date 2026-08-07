#!/bin/bash
# Tier-2 for fourc::beam_interaction#0 — what actually goes wrong when the
# BINNING STRATEGY is misconfigured, and what does not.
#
# Claimed: `zero pairs found`, or a runtime warning `bin size smaller than max
#          element size, search may miss pairs`. Neither string exists in 4C.
# Observed, on upstream beam3eb_static_contact_penalty_linpen_..._twobeamstwisting:
#   * DOMAINBOUNDINGBOX that does not contain the mesh is a HARD ERROR, not a
#     missed pair: "Node 1 in your discretization resides outside the binning
#     domain, this does not work at this point." from core/binstrategy/
#     4C_binstrategy.cpp. So the bounding box is the thing to get right first.
#   * BIN_SIZE_LOWER_BOUND raised from 1 to 1000 — far beyond any element — is
#     accepted without a word and the contact answer is unchanged: all three
#     result tests still pass. There is no "search may miss pairs" warning
#     because an oversized lower bound simply yields one big bin.
#
# NOTE this deck needs its NOX status-test XML next to it; that is why the
# fixture copies the .xml as well as the .yaml.
. "$(dirname "$0")/../_lib/preamble.sh"

DECK=beam3eb_static_contact_penalty_linpen_limitdispperiter_twobeamstwisting
BASE=$(upstream "$DECK.4C.yaml") || exit 3
XML=$(upstream "$DECK.xml")      || exit 3
cd "$TMP" || exit 3
cp "$XML" .
cp "$BASE" base.yaml
grep -q 'DOMAINBOUNDINGBOX: "-0.5 -0.5 -0.5 5.5 5.5 5.5"' base.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'BIN_SIZE_LOWER_BOUND: 1$' base.yaml \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

sed 's/DOMAINBOUNDINGBOX: "-0.5 -0.5 -0.5 5.5 5.5 5.5"/DOMAINBOUNDINGBOX: "0.0 0.0 0.0 0.1 0.1 0.1"/' base.yaml > tinybox.yaml
sed 's/BIN_SIZE_LOWER_BOUND: 1$/BIN_SIZE_LOWER_BOUND: 1000/'                                          base.yaml > bigbin.yaml

probe BASE    base.yaml
probe TINYBOX tinybox.yaml
probe BIGBIN  bigbin.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "BIGBIN_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BIGBIN.log")"
echo "BIGBIN_TESTS_WRONG=$(grep -c 'is WRONG' "$TMP/BIGBIN.log")"
grep -m1 -F "Node 1 in your discretization resides outside the binning" "$TMP/TINYBOX.log"
grep -m1 -F "4C_binstrategy.cpp" "$TMP/TINYBOX.log"
# Neither claimed string is emitted by anything.
echo "CLAIMED_ZERO_PAIRS_TEXT=$(grep -ci 'zero pairs found' "$TMP/TINYBOX.log" "$TMP/BIGBIN.log" | awk -F: '{s+=$2} END {print s+0}')"
echo "CLAIMED_BIN_SIZE_WARNING_TEXT=$(grep -ci 'search may miss pairs' "$TMP/BIGBIN.log")"
exit 0
