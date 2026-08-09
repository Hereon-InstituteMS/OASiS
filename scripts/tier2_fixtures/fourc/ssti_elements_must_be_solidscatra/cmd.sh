#!/bin/bash
# Tier-2 for fourc::ssti#1 — and a correction of a CORRECTION.
#
# The entry originally quoted 'no SCATRA discretisation found' from a
# '4C_tsi_factory.cpp'; neither exists.  It was then corrected to 'Unsupported
# solid element type!' from tsi/4C_tsi_utils.cpp.  That is a real string in the
# binary, but it is NOT what an SSTI deck with SOLID elements produces.  Two
# arms show what actually happens:
#
#   NAIVE  SOLIDSCATRA -> SOLID, TYPE token left in place.  You never reach any
#          field logic: the element line itself fails to parse with
#          "After parsing, the line still contains 'TYPE ElchElectrode'."
#          from core/io/src/4C_io_input_spec.cpp — SOLID has no TYPE key.
#
#   CLEAN  SOLIDSCATRA -> SOLID and the TYPE token removed, so the line parses.
#          NOW the field logic speaks, from src/ssti/4C_ssti_utils.cpp:
#          "ScatraStructureCloneStrategy copies scatra discretization from
#           structure discretization, but the STRUCTURE elements ... Use
#           SOLIDSCATRA, WALLSCATRA or SHELLSCATRA elements with meaningful
#           ImplType instead!"
#
# The CLEAN message is the useful one — it names the three legal element types
# and the ImplType concept.  It comes from ssti/, not from tsi/, and the tsi
# string never appears; both asserted.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssti_mono_3D_3hex8_elch_s2i_butlervolmerthermo_growthlaw.4C.yaml) || exit 3
grep -q " SOLIDSCATRA HEX8" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q " TYPE ElchElectrode" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/solidscatra.yaml"
sed 's/ SOLIDSCATRA HEX8/ SOLID HEX8/' "$BASE" > "$TMP/naive.yaml"
sed -e 's/ TYPE ElchElectrode//' -e 's/ TYPE ElchDiffCond//' "$BASE" > "$TMP/notype.yaml"
sed 's/ SOLIDSCATRA HEX8/ SOLID HEX8/' "$TMP/notype.yaml" > "$TMP/clean.yaml"

probe SOLIDSCATRA "$TMP/solidscatra.yaml"
probe NAIVE       "$TMP/naive.yaml"
probe CLEAN       "$TMP/clean.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SOLIDSCATRA.log"
# Arm 1: the element line never parses, so nothing about fields is said.
grep -m1 -F "After parsing, the line still contains 'TYPE ElchElectrode'." "$TMP/NAIVE.log"
grep -m1 -oF "4C_io_input_spec.cpp" "$TMP/NAIVE.log"
# Arm 2: the real SSTI diagnostic, naming the legal element types.
grep -m1 -F "Use SOLIDSCATRA, WALLSCATRA or SHELLSCATRA elements with meaningful ImplType instead!" "$TMP/CLEAN.log"
grep -m1 -oF "4C_ssti_utils.cpp" "$TMP/CLEAN.log"
echo "FAILS_IN_SSTI_CLONE_STRATEGY=$(grep -c 'SSTIScatraStructureCloneStrategy' "$TMP/CLEAN.log")"
# Neither the original nor the first correction is what 4C emits here.
echo "CLAIMED_NO_SCATRA_DISCRETISATION_TEXT=$(grep -ci 'no SCATRA discretisation found' "$TMP/NAIVE.log" "$TMP/CLEAN.log" | awk -F: '{s+=$2} END {print s+0}')"
echo "CLAIMED_UNSUPPORTED_SOLID_ELEMENT_TEXT=$(grep -ci 'Unsupported solid element type' "$TMP/NAIVE.log" "$TMP/CLEAN.log" | awk -F: '{s+=$2} END {print s+0}')"
echo "DIAGNOSTIC_COMES_FROM_TSI_UTILS=$(grep -c '4C_tsi_utils.cpp' "$TMP/CLEAN.log")"
exit 0
