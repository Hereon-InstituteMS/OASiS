#!/bin/bash
# Tier-2 for fourc::tsi#10 — TSI DYNAMIC/COUPALGO, checked by RUNNING 4C rather
# than by reading a schema.  Eleven copies of upstream
# tsi_lincompression_1waydisp, each with a different COUPALGO value.
#
# The seven catalogued names all get past the parser.  The four names the
# earlier catalog carried are all rejected at parse time with
# 'Could not match this input' from 4C_io_input_spec_builders.cpp — and the
# error block prints the entire legal enum on one line, which is the single
# most useful thing about it:
#
#   [!] Candidate deprecated_selection 'COUPALGO' has wrong value, possible
#   values: tsi_iterstagg|tsi_iterstagg_aitken|tsi_iterstagg_aitkenirons|
#   tsi_iterstagg_fixedrelax|tsi_monolithic|tsi_oneway|tsi_sequstagg
#
# ACCEPTED_BY_PARSER counts how many of the seven were NOT parse-rejected;
# REJECTED_BY_PARSER counts how many of the four were.  "Accepted by the
# parser" is the claim being tested — several of the seven then fail later for
# reasons of their own (this deck is tuned for tsi_oneway), which is a
# different question and is deliberately not asserted here.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_lincompression_1waydisp.4C.yaml) || exit 3
grep -q 'COUPALGO: "tsi_oneway"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

VALID="tsi_oneway tsi_sequstagg tsi_iterstagg tsi_iterstagg_aitken \
tsi_iterstagg_aitkenirons tsi_iterstagg_fixedrelax tsi_monolithic"
HISTORICAL="tsi_iterstaggaitken tsi_iterstaggaitkenirons tsi_iterstaggfixedrel monolithic"

acc=0; rej=0
for v in $VALID; do
  sed "s/COUPALGO: \"tsi_oneway\"/COUPALGO: \"$v\"/" "$BASE" > "$TMP/v.yaml"
  stdbuf -oL -eL "$BIN" "$TMP/v.yaml" "$TMP/o_v" > "$TMP/v.log" 2>&1
  grep -q "Could not match this input" "$TMP/v.log" || acc=$((acc + 1))
done
for v in $HISTORICAL; do
  sed "s/COUPALGO: \"tsi_oneway\"/COUPALGO: \"$v\"/" "$BASE" > "$TMP/h_$v.yaml"
  stdbuf -oL -eL "$BIN" "$TMP/h_$v.yaml" "$TMP/o_h" > "$TMP/h_$v.log" 2>&1
  grep -q "Could not match this input" "$TMP/h_$v.log" && rej=$((rej + 1))
done
echo "ACCEPTED_BY_PARSER=$acc/7"
echo "REJECTED_BY_PARSER=$rej/4"

# The one that used to be written 'monolithic', in full.
probe BAD_MONOLITHIC "$TMP/h_monolithic.yaml"
grep -m1 -F "Could not match this input" "$TMP/BAD_MONOLITHIC.log"
grep -m1 -oF "4C_io_input_spec_builders.cpp" "$TMP/BAD_MONOLITHIC.log"
grep -m1 -F "COUPALGO: \"monolithic\"" "$TMP/BAD_MONOLITHIC.log"
grep -m1 -F "possible values: tsi_iterstagg|tsi_iterstagg_aitken|tsi_iterstagg_aitkenirons|tsi_iterstagg_fixedrelax|tsi_monolithic|tsi_oneway|tsi_sequstagg" "$TMP/BAD_MONOLITHIC.log"
# The correctly spelled one runs the deck to completion.
cp "$BASE" "$TMP/good.yaml"
probe GOOD_ONEWAY "$TMP/good.yaml"
grep -m1 -F "processor 0 finished normally" "$TMP/GOOD_ONEWAY.log"
exit 0
