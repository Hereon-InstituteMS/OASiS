#!/bin/bash
# Tier-2 for fourc::low_mach#6 -- 4C never reports a Nusselt or a Rayleigh
# number, so "compare Nu with the literature value" needs a step the entry did
# not name.
#
# What 4C gives you is the wall heat flux, and only if you ask: CALCFLUX_BOUNDARY
# in SCALAR TRANSPORT DYNAMIC plus a SCATRA FLUX CALC ... CONDITIONS block make it
# print, per step, "Sum of all normal flux boundary integrals for scalar 0: ...".
# Nu has to be formed from that by hand.  The second arm removes the wall
# temperature difference and shows the integral responds, so it really is the
# quantity that carries the heat transfer.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }
BASE=$(upstream loma_2d_heated_chan_30x100.4C.yaml) || exit 3
link_xml "$BASE"

grep -q '  CALCFLUX_BOUNDARY: "diffusive"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^SCATRA FLUX CALC LINE CONDITIONS:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "    VAL: \[439.5\]" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/hot.yaml"
sed 's/    VAL: \[439.5\]/    VAL: [293.0]/' "$BASE" > "$TMP/isothermal.yaml"

probe HOT        "$TMP/hot.yaml"
probe ISOTHERMAL "$TMP/isothermal.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/HOT.log"
echo "HOT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/HOT.log")"
# The observable exists and is named exactly this.
grep -m1 -F "Sum of all normal flux boundary integrals for scalar 0" "$TMP/HOT.log"
grep -m1 -F "Normal fluxes at boundary 'ScaTraFluxCalc' on discretization 'scatra'" "$TMP/HOT.log"
echo "HOT_FLUX_REPORTS=$(grep -c 'Sum of all normal flux boundary integrals' "$TMP/HOT.log")"
# Remove the wall temperature difference and the same integral changes.
echo "ISOTHERMAL_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ISOTHERMAL.log")"
# Neither dimensionless group is ever printed, in either arm.
echo "NUSSELT_OR_RAYLEIGH_MENTIONS=$(grep -ciE 'nusselt|rayleigh' "$TMP/HOT.log" "$TMP/ISOTHERMAL.log" | awk -F: '{s+=$2} END {print s+0}')"
exit 0
