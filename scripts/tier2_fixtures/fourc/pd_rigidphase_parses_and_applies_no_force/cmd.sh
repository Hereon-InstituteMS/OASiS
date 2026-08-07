#!/bin/bash
# Tier-2 for fourc::particle_pd#2 — use boundaryphase, not rigidphase, for a
# rigid obstacle in a PD run.  The advice is right; the stated failure mode is
# not.
#
# Claimed:  parser error `unknown particle phase: rigidphase`, or a runtime
#           abort `rigidphase incompatible with PD interaction`.
# Observed: rigidphase is a perfectly valid particle type.  The deck parses, the
#           run completes normally, 4C says nothing at all about the phase — and
#           the rigid wall exerts EXACTLY ZERO force on the peridynamic body.
#           The PD body free-falls straight through the container: after 3000
#           steps of 8e-6 s under g = 9810 the tested particles report
#           velx = 2.354e+02, which is g*t to twelve digits, and vely = O(1e-10).
#           Eight of the deck's ten result tests fail; the damage test still
#           passes, so a reader checking only for cracks sees nothing wrong.
#
# The failure is therefore silent, not loud, which is the opposite of what the
# entry told an agent to grep for.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q 'PHASE_TO_MATERIAL_ID: "boundaryphase 1 pdphase 2"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'GRAVITY_ACCELERATION: "9810.0 0.0 0.0"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/boundary.yaml"
sed -e 's|PHASE_TO_DYNLOADBALFAC: "boundaryphase 1.0 pdphase 1.0"|PHASE_TO_DYNLOADBALFAC: "rigidphase 1.0 pdphase 1.0"|' \
    -e 's|PHASE_TO_MATERIAL_ID: "boundaryphase 1 pdphase 2"|PHASE_TO_MATERIAL_ID: "rigidphase 1 pdphase 2"|' \
    -e 's|TYPE boundaryphase|TYPE rigidphase|g' "$BASE" > "$TMP/rigid.yaml"

probe BOUNDARYPHASE "$TMP/boundary.yaml"
probe RIGIDPHASE    "$TMP/rigid.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BOUNDARYPHASE.log"
echo "BOUNDARYPHASE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BOUNDARYPHASE.log")"

# rigidphase parses: the run reaches the result-test manager.
echo "RIGIDPHASE_REACHED_RESULT_TEST=$(grep -c 'is WRONG --> actresult=\|is CORRECT, abs' "$TMP/RIGIDPHASE.log")"
echo "RIGIDPHASE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/RIGIDPHASE.log")"
grep -m1 -F "Result check failed with 8 errors out of 10 tests" "$TMP/RIGIDPHASE.log"

# ...and the PD body is in undisturbed free fall: velx == g*t == 9810*0.024.
grep -m1 -E "velx .*actresult= 2\.3544000000[0-9]+e\+02" "$TMP/RIGIDPHASE.log"
if grep -qE "velx .*actresult= 2\.3544000000[0-9]+e\+02" "$TMP/RIGIDPHASE.log"; then
  echo "PD_BODY_IS_IN_FREE_FALL=yes"
else
  echo "PD_BODY_IS_IN_FREE_FALL=no"
fi
# 4C never mentions the phase choice.
echo "CLAIMED_UNKNOWN_PARTICLE_PHASE_TEXT=$(grep -ci 'unknown particle phase' "$TMP/RIGIDPHASE.log")"
echo "CLAIMED_INCOMPATIBLE_WITH_PD_TEXT=$(grep -ci 'incompatible with PD' "$TMP/RIGIDPHASE.log")"
echo "ANY_RIGIDPHASE_WARNING=$(grep -ciE 'rigidphase.*(warn|ignor|not supported|invalid)' "$TMP/RIGIDPHASE.log")"
exit 0
