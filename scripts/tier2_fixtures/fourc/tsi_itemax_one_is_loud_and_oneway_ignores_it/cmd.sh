#!/bin/bash
# Tier-2 for fourc::tsi#5 — both halves of the ITEMAX advice, executed.
#
# TWO-WAY HALF (upstream tsi_lincompression_iterstaggdisp, COUPALGO
# tsi_iterstagg, ITEMAX 100 -> ITEMAX 1).  ITEMAX = 1 does stop before
# convergence and does change the answer, so the rule holds.  But it is NOT
# silent and it does not "look like the right answer": 4C prints
#
#     |     >>>>>> not converged in itemax steps!                            |
#
# once per coupling step — 5 times against 0 in the baseline — and the run
# still exits 0 in the sense of finishing the loop.  The size of the error is
# asserted from the result tests rather than described.
#
# ONE-WAY HALF (upstream tsi_lincompression_1waydisp, COUPALGO tsi_oneway,
# ITEMAX 1 -> ITEMAX 10).  The claim says extra iterations on a one-way problem
# "waste wall-clock — each extra iteration recomputes the second field".  They
# do not: tsi_oneway makes exactly one pass per step whatever ITEMAX says.  The
# linear-solver CALL COUNT from 4C's own TimeMonitor is identical in both runs,
# and so is every result test.  (Call counts, not seconds: a count is a
# property of the algorithm, elapsed time is a property of the afternoon.)
. "$(dirname "$0")/../_lib/preamble.sh"

TWO=$(upstream tsi_lincompression_iterstaggdisp.4C.yaml) || exit 3
ONE=$(upstream tsi_lincompression_1waydisp.4C.yaml)      || exit 3
grep -q "  ITEMAX: 100" "$TWO" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  ITEMAX: 1$"  "$ONE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$TWO" "$TMP/two_converged.yaml"
sed 's/  ITEMAX: 100/  ITEMAX: 1/' "$TWO" > "$TMP/two_itemax1.yaml"
cp "$ONE" "$TMP/one_itemax1.yaml"
sed 's/  ITEMAX: 1$/  ITEMAX: 10/' "$ONE" > "$TMP/one_itemax10.yaml"

probe TWOWAY_CONVERGED "$TMP/two_converged.yaml"
probe TWOWAY_ITEMAX1   "$TMP/two_itemax1.yaml"
probe ONEWAY_ITEMAX1   "$TMP/one_itemax1.yaml"
probe ONEWAY_ITEMAX10  "$TMP/one_itemax10.yaml"

# --- two-way -------------------------------------------------------------
grep -m1 -F "is CORRECT" "$TMP/TWOWAY_CONVERGED.log"
echo "TWOWAY_CONVERGED_COMPLAINTS=$(grep -c 'not converged in itemax steps' "$TMP/TWOWAY_CONVERGED.log")"
echo "TWOWAY_ITEMAX1_COMPLAINTS=$(grep -c 'not converged in itemax steps' "$TMP/TWOWAY_ITEMAX1.log")"
grep -m1 -oF ">>>>>> not converged in itemax steps!" "$TMP/TWOWAY_ITEMAX1.log"
# The partly-converged answer, quoted from the result test.
grep -m1 -F "temp    (T(x=0)) at node   1	 is WRONG --> actresult= 3.42210895306495843e+02, givenresult= 3.47234354640127151e+02" "$TMP/TWOWAY_ITEMAX1.log"
grep -m1 -F "dispx   (ux(x=2)) at node   2	 is WRONG --> actresult=-3.79939319465025183e-01" "$TMP/TWOWAY_ITEMAX1.log"

# --- one-way -------------------------------------------------------------
calls() {  # linear-solver Solve call count out of 4C's TimeMonitor table
  grep -m1 'Core::LinAlg::Solver:  2)   Solve' "$1" \
    | sed -n 's/.*(\([0-9][0-9]*\)).*/\1/p'
}
echo "ONEWAY_ITEMAX1_SOLVE_CALLS=$(calls "$TMP/ONEWAY_ITEMAX1.log")"
echo "ONEWAY_ITEMAX10_SOLVE_CALLS=$(calls "$TMP/ONEWAY_ITEMAX10.log")"
echo "ONEWAY_EXTRA_ITERATIONS_COST_SOLVES=$(python3 -c "
a='$(calls "$TMP/ONEWAY_ITEMAX1.log")'; b='$(calls "$TMP/ONEWAY_ITEMAX10.log")'
print('yes' if a != b else 'no')")"
grep -m1 -F "is CORRECT" "$TMP/ONEWAY_ITEMAX10.log"
grep -m1 -F "processor 0 finished normally" "$TMP/ONEWAY_ITEMAX10.log"
exit 0
