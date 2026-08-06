#!/bin/bash
# Tier-2 for fourc::sti#1 — the mechanism, and what is NOT the mechanism.
#
# The entry says temperature-dependent diffusion has to be "activated via
# appropriate material models (e.g. temperature scaling in MAT_electrode or
# MAT_soret)", otherwise the coupling is one-way.
#
# The upstream monolithic STI deck settles that: it ships
# DIFF_COEF_TEMP_SCALE_FUNCT: 0 and COND_TEMP_SCALE_FUNCT: 0 on every
# electrochemical material — i.e. NO temperature scaling of the diffusion
# coefficient anywhere — and the thermo -> scatra direction is nevertheless
# live, carried entirely by MAT_soret's SORET coefficient.  Three arms:
#
#   REF    SORET as shipped                 -> all result tests pass
#   OFF    SORET = 0 (Soret flux removed)   -> results move
#   STRONG SORET raised by 3 decades        -> results move much further, and
#                                              BOTH the scatra and the thermo
#                                              quantities move
#
# So SORET, not a temperature-scaling function, is the knob; and it is a real
# two-way coupling, because turning it off changes the SPECIES field, not only
# the temperature.  Asserted as result-test counts, which are exact.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sti_mono_3D_hex8_elch_s2i_butlervolmerpeltier_adiabatic_mortar_standard.4C.yaml) || exit 3
grep -q "      SORET: 1" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "DIFF_COEF_TEMP_SCALE_FUNCT: 0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/ref.yaml"
sed 's/      SORET: 1/      SORET: 0/'    "$BASE" > "$TMP/off.yaml"
sed 's/      SORET: 1/      SORET: 1000/' "$BASE" > "$TMP/strong.yaml"

probe REF    "$TMP/ref.yaml"
probe OFF    "$TMP/off.yaml"
probe STRONG "$TMP/strong.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/REF.log"
# No temperature-scaled diffusion coefficient anywhere in the reference deck.
echo "UPSTREAM_TEMP_SCALE_FUNCTS_ACTIVE=$(grep -c 'TEMP_SCALE_FUNCT: [1-9]' "$BASE")"
echo "REF_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/REF.log")"
echo "SORET_OFF_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/OFF.log")"
echo "SORET_STRONG_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/STRONG.log")"
# The species field moves too, so the coupling really is two-way.
echo "SORET_STRONG_MOVES_SCATRA=$(grep -c 'SCATRA   : scatra .*is WRONG' "$TMP/STRONG.log")"
echo "SORET_STRONG_MOVES_THERMO=$(grep -c 'SCATRA   : thermo .*is WRONG' "$TMP/STRONG.log")"
grep -m1 -F "is WRONG --> actresult=" "$TMP/STRONG.log"
exit 0
