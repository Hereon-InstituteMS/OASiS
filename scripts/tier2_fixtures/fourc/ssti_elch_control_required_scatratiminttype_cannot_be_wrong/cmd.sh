#!/bin/bash
# Tier-2 for fourc::ssti#3 — half right, half impossible.
#
# Claimed: "set SCATRATIMINTTYPE: 'Elch' in SSI CONTROL and include the ELCH
#          CONTROL section.  Omitting either gives plain scalar transport
#          without the electrochemical source terms — BV current is ZERO."
#
# Observed:
#   * ELCH CONTROL really is required.  Delete it and 4C aborts with
#     "Invalid type of closing equation for electric potential!" from
#     scatra_ele/4C_scatra_ele_parameter_elch.cpp.  Not a zero current: no run.
#   * SCATRATIMINTTYPE is NOT in SSI CONTROL for an SSTI problem, and you
#     cannot get it wrong.  It lives in SSTI CONTROL, defaults to Elch, and Elch
#     is its ONLY legal value — 4C prints "possible values: Elch" when you try
#     anything else.  The upstream electrochemical SSTI deck omits the key
#     entirely and runs.
#   * Adding SSI CONTROL/SCATRATIMINTTYPE to an SSTI deck parses and does
#     nothing at all.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssti_mono_3D_3hex8_elch_s2i_butlervolmerthermo_growthlaw.4C.yaml) || exit 3
grep -q "^ELCH CONTROL:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/ref.yaml"

python3 - "$BASE" "$TMP/noelchctrl.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = ('ELCH CONTROL:\n  EQUPOT: "divi"\n  DIFFCOND_FORMULATION: true\n'
       '  INITPOTCALC: true\n  COUPLE_BOUNDARY_FLUXES: false\n')
if blk not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(blk, "", 1))
PY
[ -f "$TMP/noelchctrl.yaml" ] || exit 3
sed 's/^SSTI CONTROL:$/SSTI CONTROL:\n  SCATRATIMINTTYPE: "Standard"/' "$BASE" > "$TMP/wrongtype.yaml"
sed 's/^SSTI CONTROL:$/SSI CONTROL:\n  SCATRATIMINTTYPE: "Elch"\nSSTI CONTROL:/' "$BASE" > "$TMP/ssiblock.yaml"

probe REF        "$TMP/ref.yaml"
probe NOELCHCTRL "$TMP/noelchctrl.yaml"
probe WRONGTYPE  "$TMP/wrongtype.yaml"
probe SSIBLOCK   "$TMP/ssiblock.yaml"

# The upstream electrochemical SSTI deck carries NO SCATRATIMINTTYPE at all.
echo "UPSTREAM_SETS_SCATRATIMINTTYPE=$(grep -c 'SCATRATIMINTTYPE' "$BASE")"
grep -m1 -F "processor 0 finished normally" "$TMP/REF.log"
# ELCH CONTROL is genuinely required, and it is loud.
grep -m1 -F "Invalid type of closing equation for electric potential!" "$TMP/NOELCHCTRL.log"
grep -m1 -oF "4C_scatra_ele_parameter_elch.cpp" "$TMP/NOELCHCTRL.log"
echo "NOELCHCTRL_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/NOELCHCTRL.log")"
# SCATRATIMINTTYPE lives in SSTI CONTROL and has exactly one legal value.
grep -m1 -F "'SCATRATIMINTTYPE' has wrong value, possible values: Elch" "$TMP/WRONGTYPE.log"
grep -m1 -oF "4C_io_input_spec_builders.cpp" "$TMP/WRONGTYPE.log"
# Putting it in SSI CONTROL, as the entry said, is simply inert.
grep -m1 -F "processor 0 finished normally" "$TMP/SSIBLOCK.log"
echo "SSIBLOCK_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SSIBLOCK.log")"
exit 0
