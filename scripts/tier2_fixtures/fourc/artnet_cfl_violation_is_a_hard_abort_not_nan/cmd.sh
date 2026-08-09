#!/bin/bash
# Tier-2 for fourc::arterial_network#0 — 4C checks the CFL number itself and
# aborts on it. There is no NaN and no energy message.
#
# Claimed: "dt > dx/c_max gives NaN within ~10 steps (typical 'energy not
#          conserved' message)".
# Observed, on upstream one_d_3_artery_network shortened to 50 steps: raising
# TIMESTEP from 1e-4 to 2e-3 does not produce a NaN, an energy message, or a
# gradual blow-up. The very first element evaluation throws
#     CFL number at element 0 is 2.0848661028149182
# from art_net/4C_art_net_artery_ele_calc_lin_exp.cpp, before a single result test
# runs. The number is the code's own definition, sqrt(3)*|lambda|_max*dt/dx, which
# it requires to stay below 1 -- so the usable step is dt < dx/(sqrt(3)*c_max),
# a factor sqrt(3) tighter than the claimed dt <= dx/c_max.
# The strings "energy not conserved" and "nan" never appear.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream one_d_3_artery_network.4C.yaml) || exit 3
cd "$TMP" || exit 3
grep -q 'ARTERIAL DYNAMIC:' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  TIMESTEP: 0.0001' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  NUMSTEP: 10000' "$BASE"   || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# the whole 10000-step deck takes minutes; 50 steps is enough for both arms.
sed 's/  NUMSTEP: 10000/  NUMSTEP: 50/' "$BASE" > ok.yaml
sed 's/  TIMESTEP: 0.0001/  TIMESTEP: 0.002/' ok.yaml > cfl.yaml

probe OK  ok.yaml
probe CFL cfl.yaml

echo "OK_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/OK.log")"
echo "CFL_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/CFL.log")"
grep -m1 -F "CFL number at element 0 is 2.0848661028149182" "$TMP/CFL.log"
grep -m1 -F "4C_art_net_artery_ele_calc_lin_exp.cpp" "$TMP/CFL.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/OK.log"
# The claimed failure mode does not occur.
echo "CFL_NAN_OCCURRENCES=$(grep -ciE '\bnan\b|-nan|inf\b' "$TMP/CFL.log")"
echo "CLAIMED_ENERGY_TEXT=$(grep -ci 'energy not conserved' "$TMP/CFL.log")"
echo "CFL_STEPS_BEFORE_ABORT=$(grep -c 'Vtk Files' "$TMP/CFL.log")"
exit 0
