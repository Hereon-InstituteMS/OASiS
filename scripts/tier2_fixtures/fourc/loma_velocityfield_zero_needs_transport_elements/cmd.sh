#!/bin/bash
# Tier-2 for fourc::low_mach#1 -- VELOCITYFIELD other than Navier_Stokes does not
# produce a warning; it silently switches 4C to a completely different algorithm.
#
# Claimed: the scalar-transport solver warns `VELOCITYFIELD=zero with
#          PHYSICAL_TYPE=Loma is inconsistent`, or the temperature evolves by
#          pure diffusion.
# Observed: no such warning exists.  loma_dyn switches on VELOCITYFIELD: with
#          Navier_Stokes it clones the scatra field from the fluid and runs the
#          coupled algorithm; with zero or function it takes a scatra-ONLY branch
#          that expects the user to have written a TRANSPORT ELEMENTS section by
#          hand, and aborts with "No elements in input section ---TRANSPORT
#          ELEMENTS!".  The fluid field is dropped without comment.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }
BASE=$(upstream loma_2d_heated_chan_30x100.4C.yaml) || exit 3
link_xml "$BASE"

grep -q '  VELOCITYFIELD: "Navier_Stokes"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/ns.yaml"
sed 's/  VELOCITYFIELD: "Navier_Stokes"/  VELOCITYFIELD: "zero"/'     "$BASE" > "$TMP/zero.yaml"
sed 's/  VELOCITYFIELD: "Navier_Stokes"/  VELOCITYFIELD: "function"/' "$BASE" > "$TMP/func.yaml"

probe NS   "$TMP/ns.yaml"
probe ZERO "$TMP/zero.yaml"
probe FUNC "$TMP/func.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/NS.log"
echo "NS_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NS.log")"
grep -m1 -F "No elements in input section ---TRANSPORT ELEMENTS!" "$TMP/ZERO.log"
grep -m1 -F "4C_loma_dyn.cpp" "$TMP/ZERO.log"
# 'function' takes the same non-coupled branch, so this is about the branch, not the value.
echo "FUNC_SAME_ABORT=$(grep -c 'No elements in input section ---TRANSPORT ELEMENTS!' "$TMP/FUNC.log")"
# The claimed warning is absent, and no pure-diffusion run ever happens.
echo "CLAIMED_INCONSISTENT_TEXT=$(grep -ci 'is inconsistent' "$TMP/ZERO.log")"
echo "ZERO_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/ZERO.log")"
exit 0
