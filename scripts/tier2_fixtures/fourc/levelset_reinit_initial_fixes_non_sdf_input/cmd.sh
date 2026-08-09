#!/bin/bash
# Tier-2 for fourc::level_set#4 -- REINIT_INITIAL is what turns a non-SDF initial
# field into a signed distance function, and without it the field is used as
# given, gradient error and all.
#
# The upstream elliptic-reinit deck initialises phi = 0.2*x -- a linear field with
# |grad phi| = 0.2, not 1 -- sets REINIT_INITIAL: true, and pins the six nodal
# values of the CORRECTED field, phi = x.  Setting REINIT_INITIAL: false makes
# every one of the six wrong by exactly the factor the bad gradient implies: the
# computed value is 0.2 times the pinned one, i.e. the raw initial field is what
# the solver kept.  The fixture measures that ratio rather than asserting it.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream levelset_elliptic_reinit_lin.4C.yaml) || exit 3
grep -q "  REINIT_INITIAL: true" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.2\*x"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/reinit.yaml"
sed 's/  REINIT_INITIAL: true/  REINIT_INITIAL: false/' "$BASE" > "$TMP/noreinit.yaml"

probe REINIT   "$TMP/reinit.yaml"
probe NOREINIT "$TMP/noreinit.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/REINIT.log"
echo "REINIT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/REINIT.log")"
echo "NOREINIT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOREINIT.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NOREINIT.log"
# Every failing value is the UNREINITIALISED field: act/given equals the initial
# field's gradient, so the solver kept phi = 0.2*x instead of the SDF phi = x.
echo "NOREINIT_KEPT_RAW_INITIAL_FIELD=$(sed -n 's/.*actresult=[[:space:]]*\([-0-9.eE+]*\)[[:space:]]*,[[:space:]]*givenresult=[[:space:]]*\([-0-9.eE+]*\).*/\1 \2/p' "$TMP/NOREINIT.log" | awk '{r=$1/$2; d=r-0.2; if (d<0) d=-d; if (d<1e-9) n++} END {print n+0}')"
echo "NOREINIT_WARNINGS=$(grep -ciE 'signed distance|gradient|reinit.*(skip|not)' "$TMP/NOREINIT.log")"
exit 0
