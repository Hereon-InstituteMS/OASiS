#!/bin/bash
# Tier-2 for fourc::level_set#5 -- exceeding CFL 1 on a level-set transport does
# not give O(1) overshoots.  The implicit scheme stays stable and eats the peak.
#
# Claimed: "dt > h / max(|u|) gives O(1) overshoots at the interface ... interface
#          velocity exceeds one element per step, breaking upwind stability".
# Observed: 4C's scalar transport is implicit, so there is no upwind stability
#          limit to break.  Running the upstream Gaussian-hill deck to the same
#          end time with a 10x step (10 steps instead of 100) completes with no
#          NaN, reaches its result test, and fails all four pinned values -- every
#          one of them BELOW the reference.  The error is amplitude damping, not
#          overshoot, and nothing in the log flags it.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves IFPACK_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream levelset_gaussian_hill_pbc.4C.yaml) || exit 3
grep -q "  NUMSTEP: 100" "$BASE"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  TIMESTEP: 0.01" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/cfl.yaml"
# same physical end time, ten times the step
sed -e 's/  NUMSTEP: 100/  NUMSTEP: 10/' -e 's/  TIMESTEP: 0.01/  TIMESTEP: 0.1/' "$BASE" > "$TMP/bigdt.yaml"

probe CFL   "$TMP/cfl.yaml"
probe BIGDT "$TMP/bigdt.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/CFL.log"
echo "CFL_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/CFL.log")"
echo "BIGDT_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/BIGDT.log")"
echo "BIGDT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BIGDT.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/BIGDT.log"
# Damping, not overshoot: every pinned value ends up BELOW the reference.
echo "BIGDT_VALUES_BELOW_REFERENCE=$(sed -n 's/.*actresult=[[:space:]]*\([-0-9.eE+]*\)[[:space:]]*,[[:space:]]*givenresult=[[:space:]]*\([-0-9.eE+]*\).*/\1 \2/p' "$TMP/BIGDT.log" | awk '$1 < $2 {n++} END {print n+0}')"
echo "BIGDT_VALUES_ABOVE_REFERENCE=$(sed -n 's/.*actresult=[[:space:]]*\([-0-9.eE+]*\)[[:space:]]*,[[:space:]]*givenresult=[[:space:]]*\([-0-9.eE+]*\).*/\1 \2/p' "$TMP/BIGDT.log" | awk '$1 > $2 {n++} END {print n+0}')"
echo "BIGDT_NAN=$(grep -ci 'nan' "$TMP/BIGDT.log")"
echo "BIGDT_CFL_WARNINGS=$(grep -ciE 'CFL|courant|overshoot' "$TMP/BIGDT.log")"
exit 0
