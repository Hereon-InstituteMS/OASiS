#!/bin/bash
# Tier-2 for fourc::fluid_turbulence#2 -- "CFL < 1 for explicit" is advice you
# cannot act on: the 4C fluid has no explicit time integrator.
#
# TIMEINTEGR accepts Af_Gen_Alpha, BDF2, Np_Gen_Alpha, One_Step_Theta and
# Stationary.  Every one of them is implicit, so the "explicit at CFL > 1 gives
# NaN within ~10 steps" half of the entry cannot happen in 4C at all.  The
# implicit half is real and is what the second arm measures: a 100x time step on
# the upstream channel deck runs to completion with no NaN and no warning, and
# quietly moves the pinned velocity in the fourth decimal.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE / IFPACK_XML_FILE relative to the INPUT FILE's
# directory, so a copied deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream f3_cha_8x8x8_recongradl2.4C.yaml) || exit 3
grep -q '  TIMEINTEGR: "Af_Gen_Alpha"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  TIMESTEP: 0.8" "$BASE"              || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/base.yaml"
sed 's/  TIMEINTEGR: "Af_Gen_Alpha"/  TIMEINTEGR: "Explicit_Euler"/' "$BASE" > "$TMP/explicit.yaml"
sed 's/  TIMESTEP: 0.8/  TIMESTEP: 80.0/'                            "$BASE" > "$TMP/bigdt.yaml"

probe BASE     "$TMP/base.yaml"
probe EXPLICIT "$TMP/explicit.yaml"
probe BIGDT    "$TMP/bigdt.yaml"

# There is no explicit scheme to select.
grep -m1 -F "Could not match this input" "$TMP/EXPLICIT.log"
grep -m1 -F "possible values: Af_Gen_Alpha|BDF2|Np_Gen_Alpha|One_Step_Theta|Stationary" "$TMP/EXPLICIT.log"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BASE.log")"
# A 100x step is stable, silent, and wrong.
echo "BIGDT_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/BIGDT.log")"
echo "BIGDT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BIGDT.log")"
echo "BIGDT_NAN=$(grep -ci 'nan' "$TMP/BIGDT.log")"
echo "BIGDT_CFL_WARNINGS=$(grep -ciE 'CFL|courant' "$TMP/BIGDT.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/BIGDT.log"
exit 0
