#!/bin/bash
# Tier-2 for fourc::fpsi#6 — the Beavers-Joseph slip coefficient is real, but it
# is NOT called SLIP_COEFF and it does NOT live in the coupling condition, and
# omitting it does NOT default to no-slip.
#
# Claimed:  "omitting BJS defaults to no-slip ... set the SLIP_COEFF in
#            DESIGN FPSI COUPLING SURF CONDITIONS".
# Observed, on upstream fpsi_ofsiinterface.4C.yaml:
#   SLIPCOEFF : adding SLIP_COEFF to the coupling condition is rejected at parse
#               time — "Could not match this input" from
#               core/fem/src/condition/4C_fem_condition_definition.cpp line 79.
#               The condition accepts only E / ENTITY_TYPE / NODE_SET_NAME /
#               coupling_id.
#   ALPHABJ1  : the real key is FPSI DYNAMIC/ALPHABJ, documented in
#               fpsi/4C_fpsi_input.cpp as the "Beavers-Joseph-Coefficient for
#               Slip-Boundary-Condition at Fluid-Porous-Interface (0.1-4)".  Its
#               DEFAULT IS 1.0, not no-slip: writing ALPHABJ: 1.0 explicitly
#               reproduces the untouched deck's result exactly.
#   ALPHABJ4  : and it is live — ALPHABJ: 4.0, the top of the documented range,
#               moves the node-5 displacement enough for the deck's 1e-10 result
#               test to fail.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fpsi_ofsiinterface.4C.yaml) || exit 3
grep -q '^DESIGN FPSI COUPLING SURF CONDITIONS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fpsi_condition_section_changed"; exit 3; }
grep -q '  CONVTOL: 1e-10' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fpsi_dynamic_changed"; exit 3; }
grep -q 'ALPHABJ' "$BASE" \
  && { echo "FIXTURE_ABORT=upstream_now_sets_alphabj"; exit 3; }

# The claimed key, and the real one at the top of its documented range.
CLAIMED_CONDITION_KEY='    SLIP_COEFF: 1.0'
BIG_ALPHABJ=4.0

cp "$BASE" "$TMP/default.yaml"
python3 - "$BASE" "$TMP/slipcoeff.yaml" "$CLAIMED_CONDITION_KEY" <<'PY'
import sys
t = open(sys.argv[1]).read()
head = 'DESIGN FPSI COUPLING SURF CONDITIONS:\n  - E: 7\n    coupling_id: 1\n'
assert head in t, "upstream deck no longer opens the FPSI condition list with E:7"
key = sys.argv[3]
open(sys.argv[2], "w").write(
    t.replace(head, head + (key + "\n" if key else ""), 1))
PY
sed "s/  CONVTOL: 1e-10/  CONVTOL: 1e-10\n  ALPHABJ: 1.0/"          "$BASE" > "$TMP/alphabj1.yaml"
sed "s/  CONVTOL: 1e-10/  CONVTOL: 1e-10\n  ALPHABJ: $BIG_ALPHABJ/" "$BASE" > "$TMP/alphabj4.yaml"
echo "SLIPCOEFF_IN_DECK=$(grep -c 'SLIP_COEFF' "$TMP/slipcoeff.yaml")"
grep -m1 '  ALPHABJ:' "$TMP/alphabj4.yaml" | tr -d ' ' | sed 's/^/BIG_ARM_[/;s/$/]/'

probe DEFAULT   "$TMP/default.yaml"
probe SLIPCOEFF "$TMP/slipcoeff.yaml"
probe ALPHABJ1  "$TMP/alphabj1.yaml"
probe ALPHABJ4  "$TMP/alphabj4.yaml"

grep -m1 -F "OK (2)" "$TMP/DEFAULT.log"
grep -m1 -F "Could not match this input" "$TMP/SLIPCOEFF.log"
grep -m1 -F "4C_fem_condition_definition.cpp" "$TMP/SLIPCOEFF.log"
grep -m1 -F "OK (2)" "$TMP/ALPHABJ1.log"
grep -m1 -F "Result check failed with 2 errors out of 2 tests" "$TMP/ALPHABJ4.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/ALPHABJ4.log"

# The default really is ALPHABJ = 1.0, not no-slip.
echo "EXPLICIT_ALPHABJ1_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/ALPHABJ1.log")"
echo "DEFAULT_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/DEFAULT.log")"
# ...and the coefficient is live.
echo "ALPHABJ4_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/ALPHABJ4.log")"
# The claimed key name appears nowhere in 4C's reply.
echo "SLIPCOEFF_ACCEPTED=$(grep -c 'Checking results of' "$TMP/SLIPCOEFF.log")"
exit 0
