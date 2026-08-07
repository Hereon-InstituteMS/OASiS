#!/bin/bash
# Tier-2 for fourc::low_mach#0 -- PHYSICAL_TYPE is not something you can forget
# and then debug from the flow field.  4C refuses to build the problem.
#
# Claimed: without PHYSICAL_TYPE: Loma the solver treats the flow as constant
#          density and you see "a uniformly stagnant velocity field" or "flow
#          rate ~0 even at large Ra".
# Observed: there is nothing to look at.  Setting PHYSICAL_TYPE: "Incompressible"
#          on a Low_Mach_Number_Flow problem aborts during adapter setup with
#          "Input parameter PHYSICAL_TYPE in section FLUID DYNAMIC needs to be
#          'Loma' or 'Temp_dep_water' for low-Mach-number flow!" -- which also
#          reveals the second accepted value the entry never mentioned.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }
BASE=$(upstream loma_2d_heated_chan_30x100.4C.yaml) || exit 3
link_xml "$BASE"

grep -q '  PHYSICAL_TYPE: "Loma"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/loma.yaml"
sed 's/  PHYSICAL_TYPE: "Loma"/  PHYSICAL_TYPE: "Incompressible"/' "$BASE" > "$TMP/incomp.yaml"
sed 's/  PHYSICAL_TYPE: "Loma"/  PHYSICAL_TYPE: "Temp_dep_water"/' "$BASE" > "$TMP/water.yaml"

probe LOMA   "$TMP/loma.yaml"
probe INCOMP "$TMP/incomp.yaml"
probe WATER  "$TMP/water.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/LOMA.log"
echo "LOMA_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/LOMA.log")"
grep -m1 -F "Input parameter PHYSICAL_TYPE in section FLUID DYNAMIC needs to be 'Loma' or 'Temp_dep_water' for low-Mach-number flow!" "$TMP/INCOMP.log"
grep -m1 -F "4C_adapter_fld_base_algorithm.cpp" "$TMP/INCOMP.log"
# It never reaches a time step, so there is no stagnant field to inspect.
echo "INCOMP_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/INCOMP.log")"
# The second accepted value is accepted by the same check.
echo "WATER_PASSED_PHYSICAL_TYPE_CHECK=$(grep -c 'needs to be .Loma. or .Temp_dep_water.' "$TMP/WATER.log")"
exit 0
