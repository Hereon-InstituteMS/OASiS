#!/bin/bash
# Tier-2 for fourc::input_format#9 — a FALSIFICATION.
#
# Claimed: 4C poro uses a DYNAMIC formulation even for quasi-static problems,
#          the structural momentum balance keeps the rho*a term, and a step load
#          therefore rings at the elastic wave frequency; the prescribed fix is
#          to ramp the load over ten wave traversal times.
#
# Observed: a poro deck can be fully quasi-static, and 4C insists on it being
#          consistent.  With DYNAMICTYPE Statics + FLUID TIMEINTEGR Stationary +
#          TRANSIENT_TERMS none, the SOLID DENSITY IS COMPLETELY INERT: scaling
#          it by 1000 leaves all three of the deck's result tests CORRECT at
#          their own tolerance.  There is no rho*a term to ring.
#
# Two guardrails that go with it, both raised by 4C itself:
#   * asking for transient terms next to a stationary fluid is refused, with the
#     fix named in the message;
#   * the structure and porofluid time integrators must be PAIRED -- switching
#     the structure to OneStepTheta while the fluid stays stationary is refused.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream poro_2D_quad4_const_material_nodal_orthotropic_permeability.4C.yaml) || exit 3
grep -q '  DYNAMICTYPE: "Statics"'      "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  TIMEINTEGR: "Stationary"'    "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  TRANSIENT_TERMS: "none"'     "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '      DENS: 2000$'             "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/base.yaml"
sed 's/      DENS: 2000$/      DENS: 2000000/'                 "$BASE" > "$TMP/dens1000x.yaml"
sed 's/  TRANSIENT_TERMS: "none"/  TRANSIENT_TERMS: "all"/'    "$BASE" > "$TMP/transient.yaml"
sed 's/  DYNAMICTYPE: "Statics"/  DYNAMICTYPE: "OneStepTheta"/' "$BASE" > "$TMP/ost.yaml"

# Pin that the "heavy" arm really is a thousand times heavier, so this cannot
# quietly degenerate into running the same deck twice.
echo "BASE_DECK_DENS=$(grep -m1 -o 'DENS: 2000$' "$TMP/base.yaml")"
echo "HEAVY_DECK_IS_1000X=$(grep -c 'DENS: 2000000$' "$TMP/dens1000x.yaml")"

probe BASE      "$TMP/base.yaml"
probe DENS1000X "$TMP/dens1000x.yaml"
probe TRANSIENT "$TMP/transient.yaml"
probe OST       "$TMP/ost.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_CORRECT=$(grep -c 'is CORRECT' "$TMP/BASE.log")"
# A thousandfold heavier skeleton, identical verdicts: no inertia in the balance.
echo "DENS1000X_CORRECT=$(grep -c 'is CORRECT' "$TMP/DENS1000X.log")"
echo "DENS1000X_WRONG=$(grep -c 'is WRONG' "$TMP/DENS1000X.log")"
echo "SOLID_DENSITY_CHANGED_THE_ANSWER=$([ "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/BASE.log")" = "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/DENS1000X.log")" ] && echo no || echo yes)"
# 4C refuses transient terms beside a stationary fluid, and names the fix.
grep -m1 -F "Invalid option for stationary fluid! Set 'TRANSIENT_TERMS' in section POROELASTICITY DYNAMIC to 'none'!" "$TMP/TRANSIENT.log"
grep -m1 -F "4C_poroelast_base.cpp" "$TMP/TRANSIENT.log"
# ...and it refuses a mismatched structure/fluid integrator pair.
grep -m1 -F "porous media problem is limited in functionality (only one-step-theta scheme, stationary and (af)genalpha case possible)" "$TMP/OST.log"
exit 0
