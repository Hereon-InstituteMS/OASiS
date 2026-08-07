#!/bin/bash
# Tier-2 for fourc::particle_pd#7 — FALSE as written.
#
# Claimed:  "INITRADIUS in the material must equal dx/2.  An inconsistent value
#           causes incorrect mass and volume computation.  Signal: total system
#           mass printed at startup differs ... momentum conservation fails
#           after the first step."
# Observed: INITRADIUS has no effect whatsoever on a peridynamic run.  Multiply
#           it by ten on BOTH materials of the upstream 2-D PD deck and every
#           one of the ten result tests still passes with abs(diff) exactly
#           0.00000000000000000e+00 — the answer is bit-identical, not merely
#           close.  4C never prints a total system mass at startup either.
#
#           The parameter that actually sets the mass is INITIALPARTICLESPACING
#           in PARTICLE DYNAMIC/SPH: particle mass is INITDENSITY *
#           INITIALPARTICLESPACING^KERNEL_SPACE_DIM.  Doubling it breaks eight of
#           the ten tests, which is the control arm here.  PD bond forces use
#           PERIDYNAMIC_GRID_SPACING and the horizon; PD contact gaps are
#           measured against PERIDYNAMIC_GRID_SPACING, not against radii.
#
# So the advice to keep INITRADIUS = dx/2 is harmless bookkeeping, but it is not
# what protects the mass, and an agent that "fixes" INITRADIUS to chase a mass
# error will change nothing.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "INITRADIUS: 0.5" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "INITIALPARTICLESPACING: 1.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/asdocumented.yaml"
sed 's/INITRADIUS: 0.5/INITRADIUS: 5.0/' "$BASE" > "$TMP/bigradius.yaml"
sed 's/  INITIALPARTICLESPACING: 1.0/  INITIALPARTICLESPACING: 2.0/' "$BASE" > "$TMP/bigspacing.yaml"

probe ASDOCUMENTED "$TMP/asdocumented.yaml"
probe BIGRADIUS    "$TMP/bigradius.yaml"
probe BIGSPACING   "$TMP/bigspacing.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/ASDOCUMENTED.log"
echo "ASDOCUMENTED_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ASDOCUMENTED.log")"

# A tenfold INITRADIUS changes nothing at all — bit-identical, not just within tolerance.
echo "BIGRADIUS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BIGRADIUS.log")"
echo "BIGRADIUS_EXACT_MATCHES=$(grep -c 'is CORRECT, abs(diff)= 0.00000000000000000e+00' "$TMP/BIGRADIUS.log")"
grep -m1 -F "is CORRECT, abs(diff)= 0.00000000000000000e+00" "$TMP/BIGRADIUS.log"
# Same bond count, so the horizon neighbourhood does not see the radius either.
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/BIGRADIUS.log"
# 4C never prints a total system mass to compare against an analytical one.
echo "TOTAL_MASS_PRINTED_AT_STARTUP=$(grep -ciE 'total (system )?mass' "$TMP/ASDOCUMENTED.log")"

# The control: the spacing IS what carries the mass.
echo "BIGSPACING_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BIGSPACING.log")"
if [ "$(grep -c 'is WRONG --> actresult=' "$TMP/BIGRADIUS.log")" = "0" ] \
   && [ "$(grep -c 'is WRONG --> actresult=' "$TMP/BIGSPACING.log")" -gt 0 ]; then
  echo "VERDICT: PD_INITRADIUS_AFFECTS_THE_ANSWER=no"
else
  echo "VERDICT: PD_INITRADIUS_AFFECTS_THE_ANSWER=yes"
fi
exit 0
