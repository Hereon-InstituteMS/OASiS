#!/bin/bash
# Tier-2 for fourc::fluid_turbulence#5 -- 4C does have turbulent-inflow
# generation, but it is neither Lund-Wu-Squires recycling nor a Jarrin synthetic
# eddy method, and it is not a one-line switch.
#
# The mechanism is a separate precursor domain: FLUID DYNAMIC/TURBULENT INFLOW
# with TURBULENTINFLOW: true, NUMINFLOWSTEP development steps, CANONICAL_INFLOW /
# INFLOW_HOMDIR for the sampling, and a geometric separation between the inflow
# section and the main domain declared through the FLUID TURBULENT INFLOW VOLUME
# condition.  Setting the flag alone fails twice over: first on the sampling
# description, then on the missing separation condition.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE / IFPACK_XML_FILE relative to the INPUT FILE's
# directory, so a copied deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream f3_cha_8x8x8_recongradl2.4C.yaml) || exit 3
grep -q "^FLUID DYNAMIC/TURBULENCE MODEL:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "TURBULENTINFLOW" "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/base.yaml"
FLAG='FLUID DYNAMIC/TURBULENT INFLOW:\n  TURBULENTINFLOW: true\n  CANONICAL_INFLOW: "channel_flow_of_height_2"\n  NUMINFLOWSTEP: 2\nFLUID DYNAMIC/TURBULENCE MODEL:'
HOMD='FLUID DYNAMIC/TURBULENT INFLOW:\n  TURBULENTINFLOW: true\n  CANONICAL_INFLOW: "channel_flow_of_height_2"\n  INFLOW_HOMDIR: "xz"\n  NUMINFLOWSTEP: 2\nFLUID DYNAMIC/TURBULENCE MODEL:'
sed "s|^FLUID DYNAMIC/TURBULENCE MODEL:|$FLAG|" "$BASE" > "$TMP/flagonly.yaml"
sed "s|^FLUID DYNAMIC/TURBULENCE MODEL:|$HOMD|" "$BASE" > "$TMP/withhomdir.yaml"

probe BASE       "$TMP/base.yaml"
probe FLAGONLY   "$TMP/flagonly.yaml"
probe WITHHOMDIR "$TMP/withhomdir.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
# First failure: the inflow sampling description is incomplete.
grep -m1 -F "homogeneuous plane for channel flow was specified incorrectly." "$TMP/FLAGONLY.log"
grep -m1 -F "4C_fluid_turbulence_statistics_cha.cpp" "$TMP/FLAGONLY.log"
# Second failure, once that is fixed: there is no separated inflow domain.
grep -m1 -F "Nodes with separation condition expected!" "$TMP/WITHHOMDIR.log"
grep -m1 -F "4C_fluid_discret_extractor.cpp" "$TMP/WITHHOMDIR.log"
# Neither named method appears anywhere in 4C's own vocabulary.
echo "CLAIMED_METHOD_NAMES=$(grep -ciE 'Lund|Wu-Squires|Jarrin|synthetic eddy|recycl' "$TMP/FLAGONLY.log" "$TMP/WITHHOMDIR.log" | awk -F: '{s+=$2} END {print s+0}')"
exit 0
