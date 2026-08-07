#!/bin/bash
# Tier-2 for fourc::pasi#3 — a step that suits the structure but not the particle
# contact does not show up as wobbling particles; it stops the coupling.
#
# Claimed:  "a coupled simulation that respects structural CFL but exceeds
#           particle stability shows particle positions oscillating wildly within
#           each structural step."
# Observed: you never get to see that.  4C runs a partitioned two-way PASI
#           iteration inside each step, and once the particle response stops
#           being a contraction the iteration simply fails to converge:
#             The partitioned PASI solver did not converge in ITEMAX steps!
#           from pasi/4C_pasi_partitioned_twowaycoup.cpp.  It aborts there; no
#           particle trajectory is written for the offending step and nothing in
#           the message names the particle field, the contact stiffness or the
#           time step.
#
# Both arms use the SAME structural model and the SAME contact parameters; only
# PASI DYNAMIC/TIMESTEP moves, from 5e-4 to 5e-3.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream pasi_twoway_norelax_particle_dem_1d_normalcontact_linspring_walldiscretcond.4C.yaml) || exit 3
grep -q "  TIMESTEP: 0.0005" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "COUPLING: partitioned_twowaycoup" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/smalldt.yaml"
sed 's|  TIMESTEP: 0.0005|  TIMESTEP: 0.005|' "$BASE" > "$TMP/bigdt.yaml"

probe SMALLDT "$TMP/smalldt.yaml"
probe BIGDT   "$TMP/bigdt.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SMALLDT.log"
echo "SMALLDT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SMALLDT.log")"

grep -m1 -F "The partitioned PASI solver did not converge in ITEMAX steps!" "$TMP/BIGDT.log"
grep -m1 -oE "4C_pasi_partitioned_twowaycoup\.cpp, line [0-9]+" "$TMP/BIGDT.log"
# It stops there: no result test is reached, so there is no trajectory to inspect.
echo "BIGDT_REACHED_RESULT_TEST=$(grep -c 'is WRONG --> actresult=\|is CORRECT, abs' "$TMP/BIGDT.log")"
# And the message names none of the things you would have to change.
if grep -qiE "did not converge in ITEMAX steps.*(particle|stiffness|time.?step)" "$TMP/BIGDT.log"; then
  echo "DIAGNOSTIC_NAMES_THE_PARTICLE_LIMIT=yes"
else
  echo "DIAGNOSTIC_NAMES_THE_PARTICLE_LIMIT=no"
fi
echo "CLAIMED_OSCILLATION_DIAGNOSTIC=$(grep -ciE 'oscillat' "$TMP/BIGDT.log")"
exit 0
