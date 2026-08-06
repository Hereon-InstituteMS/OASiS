#!/bin/bash
# Tier-2 for fourc::low_mach#4 -- "use the same tau definition on both fields"
# is advice that cannot be followed, because the two enums are different sets.
#
# FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION offers
# Taylor_Hughes_Zarins_Whiting_Jansen; SCALAR TRANSPORT DYNAMIC/STABILIZATION
# does not -- its list stops at Taylor_Hughes_Zarins.  Writing the fluid value
# into the scatra section is rejected, and 4C prints the whole admissible list.
# The upstream heated-channel deck itself runs with DIFFERENT definitions on the
# two fields (Whiting_Jansen for the fluid, Taylor_Hughes_Zarins for the scalar)
# and passes all four of its pinned results.
. "$(dirname "$0")/../_lib/preamble.sh"

# 4C resolves MUELU_XML_FILE relative to the INPUT FILE's directory, so a copied
# deck needs the upstream xml tree beside it.
link_xml() { ln -sfn "$(dirname "$1")/xml" "$TMP/xml" 2>/dev/null || cp -r "$(dirname "$1")/xml" "$TMP/xml"; }
BASE=$(upstream loma_2d_heated_chan_30x100.4C.yaml) || exit 3
link_xml "$BASE"

grep -q '  DEFINITION_TAU: "Taylor_Hughes_Zarins_Whiting_Jansen"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  DEFINITION_TAU: "Taylor_Hughes_Zarins"' "$BASE"               || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/mixed.yaml"
# The scatra value is the only exact-quoted "Taylor_Hughes_Zarins"; the fluid
# one carries the longer _Whiting_Jansen suffix, so this sed hits scatra only.
sed 's/  DEFINITION_TAU: "Taylor_Hughes_Zarins"/  DEFINITION_TAU: "Taylor_Hughes_Zarins_Whiting_Jansen"/' "$BASE" > "$TMP/matched.yaml"
cmp -s "$BASE" "$TMP/matched.yaml" && { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

probe MIXED   "$TMP/mixed.yaml"
probe MATCHED "$TMP/matched.yaml"

# The upstream deck deliberately mixes the two definitions and is correct.
grep -m1 -F "processor 0 finished normally" "$TMP/MIXED.log"
echo "MIXED_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/MIXED.log")"
echo "MIXED_STAIR_STEP_WARNINGS=$(grep -ciE 'discontinuity at the coupling|stair' "$TMP/MIXED.log")"
# Trying to make them match is a parse error, and 4C shows the scatra list.
grep -m1 -F "Could not match this input" "$TMP/MATCHED.log"
grep -m1 -F "possible values: Codina|Codina_wo_dt|Exact_1D|Franca_Madureira_Valentin|Franca_Madureira_Valentin_wo_dt|Franca_Valentin|Franca_Valentin_wo_dt|Numerical_Value|Shakib_Hughes_Codina|Shakib_Hughes_Codina_wo_dt|Taylor_Hughes_Zarins|Taylor_Hughes_Zarins_wo_dt|Zero" "$TMP/MATCHED.log"
echo "SCATRA_LIST_HAS_WHITING_JANSEN=$(grep -c 'possible values:.*Taylor_Hughes_Zarins_Whiting_Jansen' "$TMP/MATCHED.log")"
exit 0
