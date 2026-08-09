#!/bin/bash
# Tier-2 for fourc::ssti#5 — right effect, wrong parameter name.
#
# Claimed: "INITTEMP in the structural material MUST MATCH the initial
#          temperature field ... produces spurious thermal strain ... the
#          structure visibly contracts before any heating."
#
# Observed: the mechanism is real and the consequence is worse than "visible
# contraction", but the knob is NOT called INITTEMP.  In SSTI the thermal
# inelastic deformation gradient is MAT_InelasticDefgradLinTempIso, and its
# reference temperature is RefTemp.  INITTEMP exists in 4C only on TSI plastic /
# thermo-StVenant materials, and the upstream SSTI deck contains no INITTEMP at
# all — asserted.
#
# Move RefTemp away from the initial temperature field and the spurious thermal
# strain is large enough to invert the growth tensor:
#
#     Determinante of growth must not become negative
#     mat/4C_mat_inelastic_defgrad_factors.cpp
#
# exit 1.  So a mismatched reference temperature does not quietly bias the
# answer here — it makes the material inadmissible.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssti_mono_3D_3hex8_elch_s2i_butlervolmerthermo_growthlaw.4C.yaml) || exit 3
grep -q "MAT_InelasticDefgradLinTempIso" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "      RefTemp: 300"              "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/matched.yaml"
sed 's/      RefTemp: 300/      RefTemp: 500/' "$BASE" > "$TMP/mismatched.yaml"

probe MATCHED    "$TMP/matched.yaml"
probe MISMATCHED "$TMP/mismatched.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/MATCHED.log"
grep -m1 -F "Determinante of growth must not become negative" "$TMP/MISMATCHED.log"
grep -m1 -oF "4C_mat_inelastic_defgrad_factors.cpp" "$TMP/MISMATCHED.log"
# The parameter the entry named does not occur in a working SSTI deck.
echo "UPSTREAM_USES_INITTEMP=$(grep -c 'INITTEMP' "$BASE")"
echo "UPSTREAM_USES_REFTEMP=$(grep -c 'RefTemp' "$BASE")"
exit 0
