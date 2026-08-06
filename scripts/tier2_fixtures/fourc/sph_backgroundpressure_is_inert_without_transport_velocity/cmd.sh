#!/bin/bash
# Tier-2 for fourc::particle_sph#6 — the advice is incomplete in a way that makes
# it useless as written.
#
# Claimed:  "For free-surface problems, set BACKGROUNDPRESSURE > 0 to prevent
#           tensile instability at the free surface."
# Observed: in 4C, BACKGROUNDPRESSURE on its own does NOTHING.  It is read only
#           inside the transport-velocity branch of SPHMomentum, so unless
#           TRANSPORTVELOCITYFORMULATION is set (it defaults to
#           NoTransportVelocity, and the upstream free-surface deck leaves it
#           there) the value is never touched.  Four arms make this exact:
#
#     BASE   BACKGROUNDPRESSURE 0,     no transport velocity
#     BGP    BACKGROUNDPRESSURE 0.005, no transport velocity  -> BIT-IDENTICAL to BASE
#     TV     BACKGROUNDPRESSURE 0,     StandardTransportVelocity -> BIT-IDENTICAL to BASE
#     BOTH   BACKGROUNDPRESSURE 0.005, StandardTransportVelocity -> changes the answer
#
#   The two middle arms reproduce the deck's five result-test verdicts to the
#   last printed digit, including abs(diff) = 2.22044604925031308e-16 on density.
#   Only when both are present does anything happen.
#
# So an agent that follows the entry, sets BACKGROUNDPRESSURE > 0 on a
# free-surface deck and moves on has changed nothing at all and will believe the
# tensile instability is handled.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_1d_hydrostatic_freesurface_densityintegration_cubicspline_adami.4C.yaml) || exit 3
grep -q "      BACKGROUNDPRESSURE: 0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
echo "UPSTREAM_FREESURFACE_DECK_SETS_TRANSPORT_VELOCITY=$(grep -c 'TRANSPORTVELOCITYFORMULATION' "$BASE")"

TVLINE='  BOUNDARYPARTICLEFORMULATION: "AdamiBoundaryFormulation"'
grep -qF "$TVLINE" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/base.yaml"
sed "s|$TVLINE|$TVLINE\n  TRANSPORTVELOCITYFORMULATION: \"StandardTransportVelocity\"|" "$BASE" > "$TMP/tv.yaml"
sed 's/      BACKGROUNDPRESSURE: 0/      BACKGROUNDPRESSURE: 0.005/' "$BASE" > "$TMP/bgp.yaml"
sed 's/      BACKGROUNDPRESSURE: 0/      BACKGROUNDPRESSURE: 0.005/' "$TMP/tv.yaml" > "$TMP/both.yaml"

probe BASE "$TMP/base.yaml"
probe BGP  "$TMP/bgp.yaml"
probe TV   "$TMP/tv.yaml"
probe BOTH "$TMP/both.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
echo "BASE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BASE.log")"

# Compare the five result-test verdict lines arm by arm.
verdicts() { grep -E 'is CORRECT|is WRONG' "$1"; }
echo "BGP_VERDICTS_IDENTICAL_TO_BASE=$( [ "$(verdicts "$TMP/BGP.log")" = "$(verdicts "$TMP/BASE.log")" ] && echo yes || echo no )"
echo "TV_VERDICTS_IDENTICAL_TO_BASE=$( [ "$(verdicts "$TMP/TV.log")" = "$(verdicts "$TMP/BASE.log")" ] && echo yes || echo no )"
echo "BOTH_VERDICTS_IDENTICAL_TO_BASE=$( [ "$(verdicts "$TMP/BOTH.log")" = "$(verdicts "$TMP/BASE.log")" ] && echo yes || echo no )"
grep -m1 -F "is CORRECT, abs(diff)= 2.22044604925031308e-16" "$TMP/BGP.log"
echo "BGP_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BGP.log")"
echo "TV_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/TV.log")"

# Only the combination bites.
echo "BOTH_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BOTH.log")"
grep -m1 -F "is WRONG --> actresult= 9.76678137259415213e-01" "$TMP/BOTH.log"

if [ "$(grep -c 'is WRONG --> actresult=' "$TMP/BGP.log")" = "0" ] \
   && [ "$(grep -c 'is WRONG --> actresult=' "$TMP/BOTH.log")" -gt 0 ]; then
  echo "VERDICT: BACKGROUNDPRESSURE_ALONE_DOES_ANYTHING=no"
else
  echo "VERDICT: BACKGROUNDPRESSURE_ALONE_DOES_ANYTHING=yes"
fi
exit 0
