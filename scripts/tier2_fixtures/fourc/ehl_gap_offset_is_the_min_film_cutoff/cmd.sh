#!/bin/bash
# Tier-2 for fourc::ehl#1 — the "minimum film height cutoff" the entry asks for
# already exists in 4C, it is LUBRICATION DYNAMIC/GAP_OFFSET (default 0), and
# removing it does NOT give NaN or a singular-matrix message.
#
# Claimed:  "a near-contact event (h < 1e-12 m) gives pressure NaN or 'singular
#            Reynolds stiffness matrix'".
# Observed: upstream ehl3d_mixed.4C.yaml carries GAP_OFFSET: 0.0025 — exactly
#           the h_eff = h + h_min regularisation.  Set it to 0 and the run
#           completes every one of its 20 steps, prints no NaN, no Inf, nothing
#           containing "singular", and hands back a full set of finite
#           displacements and a finite lubricant pressure that are simply wrong:
#           5 of the deck's 7 result tests move, including the node-84 x
#           displacement flipping sign (+1.94e-2 -> -7.65e-3).
#
# That is the dangerous shape: near-contact does not announce itself.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ehl3d_mixed.4C.yaml) || exit 3
grep -q '  GAP_OFFSET: 0.0025' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_gap_offset_changed"; exit 3; }

# The pathology: the minimum-film-height offset the second arm is given.
BAD_GAP_OFFSET=0.0

cp "$BASE" "$TMP/withgap.yaml"
sed "s/  GAP_OFFSET: 0.0025/  GAP_OFFSET: $BAD_GAP_OFFSET/" "$BASE" > "$TMP/nogap.yaml"
grep -m1 '  GAP_OFFSET:' "$TMP/nogap.yaml" | tr -d ' ' | sed 's/^/NOGAP_DECK_[/;s/$/]/'

probe WITHGAP "$TMP/withgap.yaml"
probe NOGAP   "$TMP/nogap.yaml"

grep -m1 -F "OK (7)" "$TMP/WITHGAP.log"
grep -m1 -F "processor 0 finished normally" "$TMP/WITHGAP.log"
grep -m1 -F "Result check failed with 5 errors out of 7 tests" "$TMP/NOGAP.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NOGAP.log"

echo "NOGAP_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/NOGAP.log")"
# The claimed symptoms are simply not there.  'inf' only ever appears inside
# column headers, so match the bare words.
echo "CLAIMED_NAN_OR_INF=$(grep -ciE '(^|[^a-z-])(nan|inf)([^a-z-]|$)' "$TMP/NOGAP.log")"
echo "CLAIMED_SINGULAR_STIFFNESS_TEXT=$(grep -ci 'singular' "$TMP/NOGAP.log")"
echo "NOGAP_FILM_WARNINGS=$(grep -ciE 'film height|near.?contact|gap.*(zero|too small)' "$TMP/NOGAP.log")"
# The run really did reach the last time step and produce finite numbers.
echo "NOGAP_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/NOGAP.log")"
P=$(grep -m1 'pre .*at node 108' "$TMP/NOGAP.log" | grep -oE 'actresult=[ ]*[-0-9.eE+]+' | tr -d ' ' | cut -d= -f2)
echo "NOGAP_LUBRICATION_PRESSURE=$P"
echo "NOGAP_PRESSURE_IS_FINITE=$(python3 -c "import math;print('yes' if math.isfinite($P) else 'no')")"
exit 0
