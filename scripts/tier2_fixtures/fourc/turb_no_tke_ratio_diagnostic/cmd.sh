#!/bin/bash
# Tier-2 for fourc::fluid_turbulence#1 -- "compute the resolved/total TKE ratio
# from diagnostics" cannot be done: 4C emits no such diagnostic.
#
# What it does emit, and only when DUMPING_PERIOD > 0, is a plane-averaged
# statistics table in <output>.flow_statistics carrying the first- and
# second-order moments -- mean u^2, mean v^2, mean w^2, mean u*v, ... -- from
# which a resolved TKE can be assembled by hand.  There is no total, no
# subgrid-scale contribution and no ratio anywhere in the log or the file, with
# the LES model on or off.  The 80%-of-TKE rule of thumb is not checkable from
# 4C output without post-processing you write yourself.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE / IFPACK_XML_FILE relative to the INPUT FILE's
# directory, so a copied deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream f3_cha_8x8x8_recongradl2.4C.yaml) || exit 3
grep -q "  DUMPING_PERIOD: 0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  SUBGRID_DISSIPATION: true" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

sed 's/  DUMPING_PERIOD: 0/  DUMPING_PERIOD: 1/' "$BASE" > "$TMP/dump.yaml"
LESBLOCK='FLUID DYNAMIC/SUBGRID VISCOSITY:\n  C_SMAGORINSKY: 0.1\nFLUID DYNAMIC/TURBULENCE MODEL:\n  TURBULENCE_APPROACH: "CLASSICAL_LES"\n  PHYSICAL_MODEL: "Smagorinsky"'
sed "s|^FLUID DYNAMIC/TURBULENCE MODEL:|$LESBLOCK|" "$TMP/dump.yaml" > "$TMP/les.yaml"

probe DUMP "$TMP/dump.yaml"
probe LES  "$TMP/les.yaml"

# The statistics file exists and carries second-order moments...
grep -m1 -F "Statistics for turbulent incompressible channel flow (first- and second-order moments)" "$TMP/o_DUMP.flow_statistics"
echo "DUMP_HAS_SECOND_MOMENTS=$(grep -c 'mean u\^2' "$TMP/o_DUMP.flow_statistics")"
echo "DUMP_STATS_ROWS=$(grep -c '^ *-\?[0-9]' "$TMP/o_DUMP.flow_statistics")"
# ...but nothing anywhere is a TKE, resolved or total, with or without the LES model.
echo "TKE_IN_STATS_FILE=$(grep -ciE 'kinetic energy|\bTKE\b|resolved' "$TMP/o_DUMP.flow_statistics")"
echo "TKE_IN_LOG=$(grep -ciE 'kinetic energy|\bTKE\b|resolved/total' "$TMP/DUMP.log")"
echo "LES_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/LES.log")"
echo "LES_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/LES.log")"
echo "TKE_IN_LES_LOG=$(grep -ciE 'kinetic energy|\bTKE\b|resolved/total' "$TMP/LES.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/DUMP.log"
exit 0
