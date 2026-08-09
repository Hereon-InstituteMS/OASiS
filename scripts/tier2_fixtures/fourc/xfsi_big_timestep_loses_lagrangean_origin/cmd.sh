#!/bin/bash
# Tier-2 for fourc::fsi_xfem#4 -- letting the interface move too far in one step
# has a precise, quotable failure in 4C: the semi-Lagrangean reconstruction of
# newly uncovered nodes cannot find its origin.
#
# Claimed: "the cut interface jumps across several elements between time steps --
#          topology-change errors accumulate and the structure shows visibly
#          erratic trajectory".
# Observed: it does not accumulate quietly.  Three steps at the upstream dt run
#          fine; the same three steps at a 100x dt abort with
#          "<<< WARNING! Initial point for node 89 for finding the Lagrangean
#          origin not in domain! >>>" from 4C_xfem_xfluid_timeInt_std_
#          SemiLagrange.cpp -- despite the "WARNING!" wording it is a FOUR_C_THROW
#          and the run stops.  Only the step count differs between the two arms,
#          so the time step is isolated as the cause.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfsi_3D_boxes.4C.yaml) || exit 3
grep -q "  NUMSTEP: 1" "$BASE"    || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  TIMESTEP: 0.05" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/clean.yaml"
sed 's/  NUMSTEP: 1/  NUMSTEP: 3/'                        "$BASE" > "$TMP/smalldt.yaml"
sed -e 's/  NUMSTEP: 1/  NUMSTEP: 3/' -e 's/  TIMESTEP: 0.05/  TIMESTEP: 5.0/' "$BASE" > "$TMP/bigdt.yaml"

probe CLEAN   "$TMP/clean.yaml"
probe SMALLDT "$TMP/smalldt.yaml"
probe BIGDT   "$TMP/bigdt.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/CLEAN.log"
echo "CLEAN_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/CLEAN.log")"
# same number of steps, upstream dt: the cut topology keeps up
echo "SMALLDT_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SMALLDT.log")"
echo "SMALLDT_LAGRANGEAN_ABORT=$(grep -c 'Lagrangean origin not in domain' "$TMP/SMALLDT.log")"
# same number of steps, 100x dt: it does not
echo "BIGDT_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/BIGDT.log")"
echo "BIGDT_LAGRANGEAN_ABORT=$(grep -c 'Lagrangean origin not in domain' "$TMP/BIGDT.log")"
grep -m1 -F "for finding the Lagrangean origin not in domain!" "$TMP/BIGDT.log"
grep -m1 -F "4C_xfem_xfluid_timeInt_std_SemiLagrange.cpp" "$TMP/BIGDT.log"
exit 0
