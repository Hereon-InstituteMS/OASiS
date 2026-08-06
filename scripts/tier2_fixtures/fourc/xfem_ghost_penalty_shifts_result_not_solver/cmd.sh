#!/bin/bash
# Tier-2 for fourc::xfem_fluid#1 -- switching ghost-penalty stabilisation off in
# an XFEM fluid does NOT make the linear solver fail.  It quietly returns a
# different answer.
#
# Claimed: `Belos: condition number > 1e16` or `solver diverged after 0
#          iterations`.
# Observed: neither string exists.  With GHOST_PENALTY_STAB: false the run
#          reaches the result-test manager normally and three of the deck's four
#          pinned values are wrong -- the pressure by ~7e-4.  The failure is a
#          wrong number, not a solver error, which is the harder kind to notice.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfluid_ls_neumann_inflow_stab.4C.yaml) || exit 3
grep -q "GHOST_PENALTY_STAB: true" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/withgp.yaml"
sed 's/GHOST_PENALTY_STAB: true/GHOST_PENALTY_STAB: false/' "$BASE" > "$TMP/nogp.yaml"

probe WITHGP "$TMP/withgp.yaml"
probe NOGP   "$TMP/nogp.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHGP.log"
echo "WITHGP_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHGP.log")"
# The unstabilised run still gets all the way to the result test...
echo "NOGP_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOGP.log")"
echo "NOGP_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOGP.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NOGP.log"
# ...and the linear solver never complains.
echo "CLAIMED_SOLVER_TEXT=$(grep -ciE 'condition number|diverged after 0 iterations' "$TMP/NOGP.log")"
echo "SOLVER_ABORTED=$(grep -ci 'Iterative solver did not converge' "$TMP/NOGP.log")"
exit 0
