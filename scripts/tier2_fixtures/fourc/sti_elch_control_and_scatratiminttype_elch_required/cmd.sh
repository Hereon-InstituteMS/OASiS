#!/bin/bash
# Tier-2 for fourc::sti#4 — and a FALSIFICATION of its Signal.
#
# Claimed: without ELCH CONTROL and SCATRATIMINTTYPE: 'Elch', "the current
#          density does not contribute to the thermal RHS — Joule heating is
#          ZERO and temperature stays at initial conditions despite electrical
#          activity."
#
# Observed: you do not get a cold answer, you get no answer.
#
#   NOELCHCTRL  ELCH CONTROL deleted -> "Invalid type of closing equation for
#               electric potential!"  scatra_ele/4C_scatra_ele_parameter_elch.cpp
#   STANDARD    STI DYNAMIC/SCATRATIMINTTYPE switched from Elch to Standard ->
#               parses, sets up both fields, writes the t=0 output, and then
#               dies on SIGFPE (exit 136) while constructing
#               ScaTraEleCalcElchElectrodeSTIThermo — no 4C error banner, no
#               source file, nothing naming SCATRATIMINTTYPE.
#
# The STANDARD arm is the nastier one: the log looks healthy right up to the
# last written VTK file, which is asserted.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sti_mono_3D_hex8_elch_s2i_butlervolmerpeltier_adiabatic_mortar_standard.4C.yaml) || exit 3
grep -q "^ELCH CONTROL:" "$BASE"             || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  SCATRATIMINTTYPE: "Elch"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/ref.yaml"
python3 - "$BASE" "$TMP/noelchctrl.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = ('ELCH CONTROL:\n  EQUPOT: "divi"\n'
       '  DIFFCOND_FORMULATION: true\n  INITPOTCALC: true\n')
if blk not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(blk, "", 1))
PY
[ -f "$TMP/noelchctrl.yaml" ] || exit 3
sed 's/  SCATRATIMINTTYPE: "Elch"/  SCATRATIMINTTYPE: "Standard"/' "$BASE" > "$TMP/standard.yaml"

probe REF        "$TMP/ref.yaml"
probe NOELCHCTRL "$TMP/noelchctrl.yaml"
probe STANDARD   "$TMP/standard.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/REF.log"
grep -m1 -F "Invalid type of closing equation for electric potential!" "$TMP/NOELCHCTRL.log"
grep -m1 -oF "4C_scatra_ele_parameter_elch.cpp" "$TMP/NOELCHCTRL.log"
# SCATRATIMINTTYPE Standard: a raw signal, deep in the element factory.
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/STANDARD.log"
grep -m1 -oF "ScaTraEleCalcElchElectrodeSTIThermo" "$TMP/STANDARD.log"
echo "STANDARD_HAS_4C_ERROR_BANNER=$(grep -c 'PROC 0 ERROR in' "$TMP/STANDARD.log")"
echo "STANDARD_NAMES_SCATRATIMINTTYPE=$(grep -c 'SCATRATIMINTTYPE' "$TMP/STANDARD.log")"
# The log looks healthy right up to the crash: both fields wrote their t=0 output.
echo "STANDARD_WROTE_INITIAL_OUTPUT=$(grep -cF "thermo-00000" "$TMP/STANDARD.log")"
# And neither arm produces the claimed quiet, cold result.
echo "NOELCHCTRL_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/NOELCHCTRL.log")"
echo "STANDARD_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/STANDARD.log")"
exit 0
