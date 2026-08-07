#!/bin/bash
# Tier-2 for fourc::porous_media#7 — and a FALSIFICATION of both of its bounds.
#
# Claimed: BULKMODULUS in MAT_PoroDensityLawExp above 1e10 Pa "makes iterative
#          solvers stagnate"; below 1e6 Pa it is "unphysical".
#
# Observed on the upstream multiphase deck, which ships BULKMODULUS: 100 —
# four orders BELOW the claimed floor and used by 4C's own regression suite:
#
#   soft  (K = 100)   : runs, 15/15 time steps, 7/7 result tests pass
#   stiff (K = 1e11)  : runs, 15/15 time steps, no iteration-limit hit, no
#                       non-convergence; 1 of 7 result tests moves, by 1.5 %
#
# So a 1e9-fold increase past the claimed ceiling costs no extra nonlinear
# iterations at all — it does not even use more of them than the soft arm — and
# the value the suite itself ships sits far below the claimed floor.  What the
# parameter does do is change the answer, which is asserted as a real observation
# (the result-test verdict), not as a stability warning.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream porofluid_pressure_based_2D_quad4.4C.yaml) || exit 3
ln -s "$(dirname "$BASE")/xml" "$TMP/xml"
grep -q "      BULKMODULUS: 100" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/soft.yaml"
sed 's/      BULKMODULUS: 100/      BULKMODULUS: 1e+11/' "$BASE" > "$TMP/stiff.yaml"

probe SOFT  "$TMP/soft.yaml"
probe STIFF "$TMP/stiff.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SOFT.log"
# The shipped value is 100 Pa — below the claimed 1e6 floor — and passes.
echo "SOFT_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SOFT.log")"
# Above the claimed 1e10 ceiling: still every time step, still converging.
echo "SOFT_TIME_STEPS=$(grep -c 'PORO MULTIPHASE FLUID SOLVER' "$TMP/SOFT.log")"
echo "STIFF_TIME_STEPS=$(grep -c 'PORO MULTIPHASE FLUID SOLVER' "$TMP/STIFF.log")"
echo "STIFF_HIT_ITERATION_LIMIT=$(grep -c '^|   50/ 50' "$TMP/STIFF.log")"
echo "STIFF_NONCONVERGENCE_MESSAGES=$(grep -ciE 'not converged|did not converge' "$TMP/STIFF.log")"
S_IT=$(grep -cE '^\|  *[0-9]+/ 50' "$TMP/SOFT.log")
K_IT=$(grep -cE '^\|  *[0-9]+/ 50' "$TMP/STIFF.log")
echo "SOFT_NONLINEAR_ITERATIONS=$S_IT"
echo "STIFF_NONLINEAR_ITERATIONS=$K_IT"
if [ "$K_IT" -gt "$S_IT" ]; then
  echo "VERDICT: HIGH_BULKMODULUS_COSTS_MORE_ITERATIONS=yes"
else
  echo "VERDICT: HIGH_BULKMODULUS_COSTS_MORE_ITERATIONS=no"
fi
# It does change the answer, though — that part is real and is what the entry
# should warn about.
grep -m1 -F "Result check failed with 1 errors out of 7 tests" "$TMP/STIFF.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/STIFF.log"
exit 0
