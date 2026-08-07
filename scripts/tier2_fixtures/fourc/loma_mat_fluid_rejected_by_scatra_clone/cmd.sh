#!/bin/bash
# Tier-2 for fourc::low_mach#2 -- MAT_fluid in a LOMA deck is rejected, but by
# the scatra cloning strategy, and the message never mentions PHYSICAL_TYPE.
#
# Claimed: 4C aborts with `material MAT_fluid incompatible with PHYSICAL_TYPE
#          Loma`, OR the results show constant density.
# Observed: no such string.  Because a LOMA run clones its scalar-transport
#          discretisation from the fluid one, the material is validated by
#          ScatraFluidCloneStrategy, which throws "Material with ID 1 is not
#          admissible for scalar transport elements" from
#          4C_scatra_utils_clonestrategy.cpp.  There is no second, quieter
#          outcome: you never get constant-density results to look at.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }
BASE=$(upstream loma_2d_heated_chan_30x100.4C.yaml) || exit 3
link_xml "$BASE"

grep -q "    MAT_sutherland:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/suth.yaml"
python3 - "$BASE" "$TMP/matfluid.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = """  - MAT: 1
    MAT_sutherland:
      REFVISC: 0.01178
      REFTEMP: 293
      SUTHTEMP: 110.4
      SHC: 1004.5
      PRANUM: 1
      THERMPRESS: 98100
      GASCON: 287"""
new = """  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: 0.01178
      DENSITY: 1.1662
      GAMMA: 1"""
assert old in t, "upstream deck no longer carries the Sutherland material"
open(sys.argv[2], "w").write(t.replace(old, new))
PY

probe SUTH     "$TMP/suth.yaml"
probe MATFLUID "$TMP/matfluid.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SUTH.log"
echo "SUTH_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SUTH.log")"
grep -m1 -F "Material with ID 1 is not admissible for scalar transport elements" "$TMP/MATFLUID.log"
grep -m1 -F "4C_scatra_utils_clonestrategy.cpp" "$TMP/MATFLUID.log"
echo "MATFLUID_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/MATFLUID.log")"
echo "CLAIMED_INCOMPATIBLE_TEXT=$(grep -ci 'incompatible with PHYSICAL_TYPE' "$TMP/MATFLUID.log")"
echo "ABORT_LINE_MENTIONS_LOMA=$(grep -F 'Material with ID 1 is not admissible' "$TMP/MATFLUID.log" | grep -ci 'Loma')"
echo "ABORT_LINE_MENTIONS_MAT_FLUID=$(grep -F 'Material with ID 1 is not admissible' "$TMP/MATFLUID.log" | grep -ci 'MAT_fluid')"
exit 0
