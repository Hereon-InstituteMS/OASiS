#!/bin/bash
# Tier-2 for fourc::reduced_airways#1 — drop the driving boundary condition at
# the tracheal node and 4C solves a perfectly converged nothing.
#
# Claimed: "simulation completes but the flow / volume result fields are
#          uniformly zero; OR runtime warning `no DESIGN POINT 1D DBC defined at
#          trachea node`".
# Observed, on upstream red_airway_one_acinus_NeoHookean: the first half is exact
# and the second half does not exist. Removing the E:1 entry from
# "DESIGN NODE Reduced D AIRWAYS PRESCRIBED CONDITIONS" leaves a deck that parses,
# runs all 500 steps, and prints the same line every step:
#     |Pressure|_max:  0.000E+00 			 |Q|_max:  0.000E+00
# There is no warning of any kind about a missing boundary condition — the
# airway-specific diagnostic counter comes out 0. Only the deck's own result test
# notices, with actresult exactly 0 against an expected 521.378.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream red_airway_one_acinus_NeoHookean.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" base.yaml

python3 - <<'PY'
t = open('base.yaml').read()
drive = '''  - E: 1
    VAL: [1]
    curve: [1, null]
'''
assert drive in t, "upstream deck no longer carries the DNODE 1 tracheal drive"
open('nobc.yaml', 'w').write(t.replace(drive, ''))
PY

probe BASE base.yaml
probe NOBC nobc.yaml

echo "BASE_TESTS_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
echo "NOBC_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOBC.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "is WRONG --> actresult= 0.00000000000000000e+00" "$TMP/NOBC.log"
# Every single step of the undriven run reports an identically zero field.
echo "NOBC_DISTINCT_FIELD_MAXIMA=$(grep -c 'Pressure|_max' "$TMP/NOBC.log" > /dev/null; grep 'Pressure|_max' "$TMP/NOBC.log" | sort -u | wc -l)"
grep -m1 -F '|Pressure|_max:  0.000E+00' "$TMP/NOBC.log"
# and 4C never says the boundary condition is missing.
echo "MISSING_BC_WARNINGS=$(grep -ciE 'boundary condition|prescribed condition|trachea|DBC' "$TMP/NOBC.log")"
echo "CLAIMED_TRACHEA_TEXT=$(grep -ci 'no DESIGN POINT 1D DBC' "$TMP/NOBC.log")"
exit 0
