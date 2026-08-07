#!/bin/bash
# Tier-2 for fourc::fluid_turbulence#4 -- there is no TURBULENCE_MODEL key, and
# naming an SGS model is not enough to switch one on.
#
# Turning on Smagorinsky in 4C takes THREE settings, and any one of them missing
# leaves the model completely inert with no warning:
#   FLUID DYNAMIC/TURBULENCE MODEL:  TURBULENCE_APPROACH: CLASSICAL_LES
#   FLUID DYNAMIC/TURBULENCE MODEL:  PHYSICAL_MODEL: Smagorinsky
#   FLUID DYNAMIC/SUBGRID VISCOSITY: C_SMAGORINSKY: <nonzero>   (default is 0)
#
# The arms below hold that: the claimed key is unused input; PHYSICAL_MODEL plus
# a non-zero constant but the default approach reproduces the no-model answer
# bit for bit; adding CLASSICAL_LES finally moves it.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE / IFPACK_XML_FILE relative to the INPUT FILE's
# directory, so a copied deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }

BASE=$(upstream f3_cha_8x8x8_recongradl2.4C.yaml) || exit 3
grep -q "^FLUID DYNAMIC/TURBULENCE MODEL:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "TURBULENCE_APPROACH" "$BASE" && { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
link_xml "$BASE"

cp "$BASE" "$TMP/nomodel.yaml"
sed 's|^FLUID DYNAMIC/TURBULENCE MODEL:|FLUID DYNAMIC/TURBULENCE MODEL:\n  TURBULENCE_MODEL: "Smagorinsky"|' "$BASE" > "$TMP/claimedkey.yaml"
HALF='FLUID DYNAMIC/SUBGRID VISCOSITY:\n  C_SMAGORINSKY: 0.1\nFLUID DYNAMIC/TURBULENCE MODEL:\n  PHYSICAL_MODEL: "Smagorinsky"'
FULL='FLUID DYNAMIC/SUBGRID VISCOSITY:\n  C_SMAGORINSKY: 0.1\nFLUID DYNAMIC/TURBULENCE MODEL:\n  TURBULENCE_APPROACH: "CLASSICAL_LES"\n  PHYSICAL_MODEL: "Smagorinsky"'
sed "s|^FLUID DYNAMIC/TURBULENCE MODEL:|$HALF|" "$BASE" > "$TMP/halfon.yaml"
sed "s|^FLUID DYNAMIC/TURBULENCE MODEL:|$FULL|" "$BASE" > "$TMP/fullon.yaml"

probe NOMODEL    "$TMP/nomodel.yaml"
probe CLAIMEDKEY "$TMP/claimedkey.yaml"
probe HALFON     "$TMP/halfon.yaml"
probe FULLON     "$TMP/fullon.yaml"

# The key the entry used does not exist.
grep -m1 -F "Could not match this input" "$TMP/CLAIMEDKEY.log"
grep -m1 -F "TURBULENCE_MODEL: \"Smagorinsky\"" "$TMP/CLAIMEDKEY.log"
grep -m1 -F "Defaulted deprecated_selection 'PHYSICAL_MODEL'" "$TMP/CLAIMEDKEY.log"
grep -m1 -F "processor 0 finished normally" "$TMP/NOMODEL.log"
echo "NOMODEL_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOMODEL.log")"
# PHYSICAL_MODEL + a real constant, but the default approach: silently inert.
echo "HALFON_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/HALFON.log")"
echo "HALFON_WARNINGS=$(grep -ciE 'turbulence model.*(ignor|inactive|no effect)' "$TMP/HALFON.log")"
# Add CLASSICAL_LES and the same deck finally changes.
echo "FULLON_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/FULLON.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/FULLON.log"
exit 0
