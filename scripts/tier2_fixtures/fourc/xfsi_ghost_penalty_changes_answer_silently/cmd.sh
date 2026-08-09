#!/bin/bash
# Tier-2 for fourc::fsi_xfem#1 -- ghost penalty in XFSI is an accuracy switch
# that fails silently, not a solver-robustness switch that announces itself.
#
# Claimed: cut elements with tiny volume fractions make the matrix singular or
#          ill-conditioned, cond(K) > 1e16 and direct LU fails.
# Observed: on the upstream monolithic XFSI deck the direct solver (UMFPACK) has
#          no trouble at all.  Turning ghost penalty off, or leaving it on with
#          GHOST_PENALTY_FAC: 0.0, lets the run finish its Newton loop and reach
#          the result test -- where six of seven pinned values are wrong.  No
#          condition number is ever reported and no factorisation fails.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfsi_3D_boxes.4C.yaml) || exit 3
grep -q "  GHOST_PENALTY_STAB: true" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  GHOST_PENALTY_FAC: 0.05" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'SOLVER: "UMFPACK"' "$BASE"          || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/withgp.yaml"
sed 's/  GHOST_PENALTY_STAB: true/  GHOST_PENALTY_STAB: false/' "$BASE" > "$TMP/nogp.yaml"
sed 's/  GHOST_PENALTY_FAC: 0.05/  GHOST_PENALTY_FAC: 0.0/'     "$BASE" > "$TMP/zerofac.yaml"

probe WITHGP  "$TMP/withgp.yaml"
probe NOGP    "$TMP/nogp.yaml"
probe ZEROFAC "$TMP/zerofac.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHGP.log"
echo "WITHGP_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHGP.log")"
echo "NOGP_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOGP.log")"
echo "NOGP_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOGP.log")"
echo "ZEROFAC_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ZEROFAC.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NOGP.log"
echo "CLAIMED_CONDITIONING_TEXT=$(grep -ciE 'condition number|singular matrix|LU fail' "$TMP/NOGP.log")"
echo "LINEAR_SOLVER_FAILED=$(grep -ciE 'did not converge|zero pivot|factorization failed' "$TMP/NOGP.log")"
exit 0
