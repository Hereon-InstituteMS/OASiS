#!/bin/bash
# Tier-2 for fourc::fpsi#3 — INITPOROSITY must be in the open interval (0,1), but
# the two endpoints behave completely differently and neither prints the NaN the
# entry promised.
#
# Claimed:  "INITPOROSITY = 0 (no pores) or INITPOROSITY = 1 (all pores, no
#            skeleton) triggers DIVISION BY ZERO in Darcy flow (NaN appears in
#            fluid velocity at the first iteration)".
# Observed, on upstream fpsi_ofsiinterface.4C.yaml (MAT_StructPoro,
# INITPOROSITY 0.5, two result tests):
#   0.0 : the process is KILLED by SIGFPE, "Invalid floating point operation",
#         inside Mat::PAR::PoroLawNeoHooke::compute_porosity — in the porous
#         constitutive law, not in Darcy flow.  Shell status 136, no 4C error
#         line, no NaN ever printed because nothing gets a chance to print.
#   1.0 : no crash at all.  The run completes, and simply returns a different
#         answer, caught only by the deck's own result tests (2 of 2).
#
# The dangerous endpoint is therefore 1.0, not 0.0: one is a hard stop, the other
# is silent.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fpsi_ofsiinterface.4C.yaml) || exit 3
grep -q '      INITPOROSITY: 0.5' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_initporosity_changed"; exit 3; }

# The two endpoint pathologies.
POROSITY_ZERO=0.0
POROSITY_ONE=1.0

cp "$BASE" "$TMP/mid.yaml"
sed "s/      INITPOROSITY: 0.5/      INITPOROSITY: $POROSITY_ZERO/" "$BASE" > "$TMP/zero.yaml"
sed "s/      INITPOROSITY: 0.5/      INITPOROSITY: $POROSITY_ONE/"  "$BASE" > "$TMP/one.yaml"
grep -m1 '      INITPOROSITY:' "$TMP/zero.yaml" | tr -d ' ' | sed 's/^/ZERO_ARM_/'
grep -m1 '      INITPOROSITY:' "$TMP/one.yaml"  | tr -d ' ' | sed 's/^/ONE_ARM_/'

probe MID  "$TMP/mid.yaml"
probe ZERO "$TMP/zero.yaml"
probe ONE  "$TMP/one.yaml"

grep -m1 -F "OK (2)" "$TMP/MID.log"
grep -m1 -F "processor 0 finished normally" "$TMP/MID.log"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/ZERO.log"
grep -m1 -F "Signal code: Invalid floating point operation (7)" "$TMP/ZERO.log"
grep -m1 -oF "PoroLawNeoHooke" "$TMP/ZERO.log"
grep -m1 -F "Result check failed with 2 errors out of 2 tests" "$TMP/ONE.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/ONE.log"

# Endpoint 0: killed before anything can be reported.
echo "ZERO_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/ZERO.log")"
echo "ZERO_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/ZERO.log")"
echo "CLAIMED_NAN_AT_POROSITY_ZERO=$(grep -ciE '(^|[^a-z-])(nan|inf)([^a-z-]|$)' "$TMP/ZERO.log")"
# Endpoint 1: completes, silently different.
echo "ONE_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/ONE.log")"
echo "ONE_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/ONE.log")"
echo "ONE_POROSITY_WARNINGS=$(grep -ciE 'porosity.*(range|invalid|zero|one)|no skeleton' "$TMP/ONE.log")"
exit 0
