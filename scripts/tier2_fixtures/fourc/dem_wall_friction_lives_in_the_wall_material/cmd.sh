#!/bin/bash
# Tier-2 for fourc::particle_dem#8 — particle-to-wall friction comes from
# MAT_ParticleWallDEM, and the two friction knobs validate OPPOSITELY.
#
# Claimed: zero in the PARTICLE DYNAMIC/DEM section is refused; the same zero in
#          MAT_ParticleWallDEM is accepted silently and changes the answer.
#          Pointing the condition's MAT at any other material gives a message
#          naming a C++ class.
#
# T2_MUTATE=1 removes every edit; the wall material keeps its friction, the
# answer does not move, WALLMAT_SILENTLY_CHANGES_THE_ANSWER goes to no and the
# fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

ROL=$(upstream particle_dem_2d_rollingcontact_coulomb.4C.yaml) || exit 3
TW=$(upstream particle_dem_3d_tangentialcontact_linspringdamp_walldiscretcond.4C.yaml) || exit 3
grep -q "MAT_ParticleWallDEM" "$ROL" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$ROL" "$TW" "$TMP" "$MUTATE" <<'PY'
import re, sys
rol, tw = open(sys.argv[1]).read(), open(sys.argv[2]).read()
tmp, mutate = sys.argv[3], sys.argv[4] == "1"
head, sep, rest = rol.partition("FUNCT1:")
assert "      FRICT_COEFF_TANG: 0.8" in rest, "wall material no longer sets tangential friction"
# zero the WALL material's friction only; the DEM section keeps its own value
z = head + sep + rest.replace("      FRICT_COEFF_TANG: 0.8", "      FRICT_COEFF_TANG: 0.0")
open(tmp + "/wallzero.yaml", "w").write(rol if mutate else z)
# point the wall condition's MAT at the DEM PARTICLE material instead
bad = re.sub(r"(DESIGN SURFACE PARTICLE WALL:.*?MAT: )\d+", r"\g<1>1", tw, flags=re.S)
open(tmp + "/wrongmat.yaml", "w").write(tw if mutate else bad)
PY

probe WALLZERO "$TMP/wallzero.yaml"
probe WRONGMAT "$TMP/wrongmat.yaml"
probe ROLBASE  "$ROL"

# Zero wall friction is ACCEPTED: no abort from the DEM stack at all ...
echo "WALLZERO_DEM_ABORTS=$(grep -c 'invalid input parameter FRICT_COEFF' "$TMP/WALLZERO.log")"
echo "WALLZERO_WARNINGS=$(grep -ci 'frict\|wall material' "$TMP/WALLZERO.log")"
# ... it runs the whole time loop ...
grep -m1 -F "Checking results of" "$TMP/WALLZERO.log" && echo "WALLZERO_REACHED_RESULT_TESTS=yes"
# ... and the only trace is that the answer moved.
echo "WALLZERO_VERDICTS_WRONG=$(grep -c 'is WRONG' "$TMP/WALLZERO.log")"
echo "WALLMAT_SILENTLY_CHANGES_THE_ANSWER=$([ "$(grep -c 'is WRONG' "$TMP/WALLZERO.log")" -gt 0 ] && echo yes || echo no)"
# The same deck untouched passes, so the difference is the wall material alone.
grep -m1 -E "^OK \(" "$TMP/ROLBASE.log" && echo "ROLBASE_PASSES=yes"
# A wall MAT pointing at the wrong material names a C++ class, not the key.
grep -m1 -F "cast to Mat::ParticleWallMaterialDEM failed!" "$TMP/WRONGMAT.log"   && echo "WRONG_WALLMAT_NAMES_A_CLASS=yes" || echo "WRONG_WALLMAT_NAMES_A_CLASS=no"
echo "WRONG_WALLMAT_NAMES_THE_INPUT_KEY=$(grep -c 'DESIGN SURFACE PARTICLE WALL\|PARTICLE_WALL_MAT' "$TMP/WRONGMAT.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# MAT: -1 -- no wall material at all -- is legal upstream for pure normal contact.
echo "UPSTREAM_DECKS_WITH_MAT_MINUS_ONE=$(grep -l 'MAT: -1' "$DECKS"/pasi_*particle_dem*.4C.yaml 2>/dev/null | wc -l)"
exit 0
