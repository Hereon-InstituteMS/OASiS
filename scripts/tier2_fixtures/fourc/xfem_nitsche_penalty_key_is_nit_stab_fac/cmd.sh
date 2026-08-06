#!/bin/bash
# Tier-2 for fourc::xfem_fluid#5 -- the knob is not called
# NITSCHE_PENALTY_PARAMETER, and on a Nitsche-coupled XFEM fluid its value does
# not do what the entry claimed.
#
# Claimed key:  NITSCHE_PENALTY_PARAMETER, "typical values O(10)--O(100)".
# Real key:     NIT_STAB_FAC in XFLUID DYNAMIC/STABILIZATION (default 35), with
#               NIT_STAB_FAC_TANG for the tangential term.  4C prints the whole
#               candidate key list when you get the name wrong.
# Claimed symptoms: too low -> interface jump violation > 5%; too high ->
#               condition number > 1e14 and Newton stalls.
# Observed:     on the upstream Nitsche weak-Dirichlet channel deck, sweeping the
#               real key over fifteen orders of magnitude (1e-3 .. 1e12) leaves
#               all four pinned values exact to 1e-13.  Nitsche is consistent, so
#               the penalty does not perturb a field the discretisation can
#               reproduce; neither extreme stalls anything.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfluid_channel_constWDBC_straightCut_hex20_EOS_GP_2ndGP.4C.yaml) || exit 3
grep -q "  NIT_STAB_FAC: 30" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/default.yaml"
sed 's/  NIT_STAB_FAC: 30/  NIT_STAB_FAC: 0.001/'  "$BASE" > "$TMP/lo.yaml"
sed 's/  NIT_STAB_FAC: 30/  NIT_STAB_FAC: 1.0e12/' "$BASE" > "$TMP/hi.yaml"
sed 's/  NIT_STAB_FAC: 30/  NITSCHE_PENALTY_PARAMETER: 30/' "$BASE" > "$TMP/claimed.yaml"

probe DEFAULT "$TMP/default.yaml"
probe LO      "$TMP/lo.yaml"
probe HI      "$TMP/hi.yaml"
probe CLAIMED "$TMP/claimed.yaml"

# The claimed key name is not accepted, and 4C names the real one in the same block.
grep -m1 -F "Could not match this input" "$TMP/CLAIMED.log"
grep -m1 -F "NITSCHE_PENALTY_PARAMETER: 30" "$TMP/CLAIMED.log"
grep -m1 -F "Defaulted parameter 'NIT_STAB_FAC'" "$TMP/CLAIMED.log"
# The real key, swept over 15 decades, changes nothing on this patch test.
echo "LO_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/LO.log")"
echo "HI_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/HI.log")"
echo "HI_NEWTON_STALLED=$(grep -ciE 'did not converge|Newton unconverged' "$TMP/HI.log")"
echo "CLAIMED_CONDITION_NUMBER_TEXT=$(grep -ci 'condition number' "$TMP/HI.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/HI.log"
exit 0
