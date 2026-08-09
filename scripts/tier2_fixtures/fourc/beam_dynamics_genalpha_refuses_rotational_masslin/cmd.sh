#!/bin/bash
# Tier-2 for fourc::beams#5 — "use GenAlphaLieGroup for beam dynamics" is right,
# but you will never see the symptom the entry described.
#
# Claimed:  a rotating beam under standard GenAlpha "shows monotonically growing
#           angular momentum (energy drift)"; switch to GenAlphaLieGroup and it
#           is conserved. That presumes the GenAlpha run happens.
# Observed: it does not. Classical GenAlpha refuses the setting beam3r needs:
#
#     MASSLIN=ml_rotations is not supported by classical GenAlpha!
#     Choose GenAlphaLieGroup instead!
#     .../structure_new/src/implicit/4C_structure_new_impl_genalpha.cpp
#
#   and dropping MASSLIN to 'none' so GenAlpha will accept it does not give a
#   drifting solution either — beam3r's inertia routine then dereferences the
#   element-internal state it was never given and the process dies of SIGSEGV
#   inside Beam3r::calc_inertia_force_and_mass_matrix, with no 4C diagnostic at
#   all. Neither arm reaches step 1, so there is no energy history to look at.
#
# Upstream deck: beam3r_line3_genalpha_liegroup_3Dtwistmoment — 64 BEAM3R LINE3
# elements, 20 steps, tip moment, DYNAMICTYPE GenAlphaLieGroup + MASSLIN rotations.
. "$(dirname "$0")/../_lib/preamble.sh"
ulimit -c 0   # the third arm dies of SIGSEGV; do not leave core files behind

BASE=$(upstream beam3r_line3_genalpha_liegroup_3Dtwistmoment.4C.yaml) || exit 3
grep -q 'DYNAMICTYPE: "GenAlphaLieGroup"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'MASSLIN: "rotations"'            "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/liegroup.yaml"
sed 's/DYNAMICTYPE: "GenAlphaLieGroup"/DYNAMICTYPE: "GenAlpha"/' "$BASE" > "$TMP/genalpha.yaml"
sed -e 's/DYNAMICTYPE: "GenAlphaLieGroup"/DYNAMICTYPE: "GenAlpha"/' \
    -e 's/MASSLIN: "rotations"/MASSLIN: "none"/' "$BASE" > "$TMP/genalpha_nomasslin.yaml"

probe LIEGROUP  "$TMP/liegroup.yaml"
probe GENALPHA  "$TMP/genalpha.yaml"
probe GANOMASS  "$TMP/genalpha_nomasslin.yaml"

# The Lie-group arm is the control: it runs all 20 steps and its result test passes.
grep -m1 -F "processor 0 finished normally" "$TMP/LIEGROUP.log"
echo "LIEGROUP_STEPS=$(grep -c 'Finalised step' "$TMP/LIEGROUP.log")"
echo "LIEGROUP_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/LIEGROUP.log")"

# Arm 2: classical GenAlpha rejects the rotational mass linearisation outright.
grep -m1 -F "MASSLIN=ml_rotations is not supported by classical GenAlpha! Choose GenAlphaLieGroup instead!" "$TMP/GENALPHA.log"
grep -m1 -F "4C_structure_new_impl_genalpha.cpp" "$TMP/GENALPHA.log"

# Arm 3: GenAlpha with MASSLIN none is accepted by the input layer and then
# segfaults inside the beam element. No diagnostic is printed.
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/GANOMASS.log"
grep -m1 -F "calc_inertia_force_and_mass_matrix" "$TMP/GANOMASS.log"
echo "GANOMASS_4C_DIAGNOSTICS=$(grep -c 'PROC 0 ERROR' "$TMP/GANOMASS.log")"

# Neither GenAlpha arm reaches a single step, so the claimed angular-momentum
# drift is not an observable of this code.
echo "GENALPHA_STEPS=$(grep -c 'Finalised step' "$TMP/GENALPHA.log")"
echo "GANOMASS_STEPS=$(grep -c 'Finalised step' "$TMP/GANOMASS.log")"
exit 0
