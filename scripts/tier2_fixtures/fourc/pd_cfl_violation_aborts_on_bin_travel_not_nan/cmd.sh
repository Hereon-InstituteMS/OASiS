#!/bin/bash
# Tier-2 for fourc::particle_pd#6 — an over-large explicit time step really does
# destroy a PD run, but you never get to see the NaN.
#
# Claimed:  "kinetic energy grows monotonically with each step, particle
#           velocities saturate at NaN within a few hundred steps; or 4C aborts
#           with `non-finite velocity at particle X`".
# Observed: 4C aborts long before anything becomes non-finite, in the particle
#           algorithm's own bookkeeping:
#             a particle of phase 'pdphase' traveled more than one bin on this processor!
#           from particle/src/algorithm/4C_particle_algorithm.cpp.  The message
#           names the PHASE, not the particle, prints no velocity and no step
#           index, and says nothing about stability or the CFL condition — so an
#           agent grepping for "nan" or "velocity" learns nothing.  There is no
#           `non-finite velocity at particle X` string in 4C at all, and the log
#           contains no NaN.
#
# The upstream deck runs at dt = 8e-6; the bad arm uses 1.6e-4.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "TIMESTEP: 8.0e-6" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/stable.yaml"
sed 's|TIMESTEP: 8.0e-6|TIMESTEP: 1.6e-4|' "$BASE" > "$TMP/toolarge.yaml"

probe STABLEDT  "$TMP/stable.yaml"
probe TOOLARGE  "$TMP/toolarge.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/STABLEDT.log"
echo "STABLEDT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/STABLEDT.log")"

grep -m1 -F "a particle of phase 'pdphase' traveled more than one bin on this processor!" "$TMP/TOOLARGE.log"
grep -m1 -oE "4C_particle_algorithm\.cpp, line [0-9]+" "$TMP/TOOLARGE.log"
# The abort fires early, not "within a few hundred steps" of NaN growth.
echo "TOOLARGE_STEPS_BEFORE_ABORT=$(grep -c 'pd_neighbor_pairs in peridynamic evaluation' "$TMP/TOOLARGE.log")"
# Nothing non-finite is ever printed, and the claimed message does not exist.
echo "CLAIMED_NONFINITE_VELOCITY_TEXT=$(grep -ci 'non-finite velocity' "$TMP/TOOLARGE.log")"
echo "NAN_IN_LOG=$(grep -c -iE '\bnan\b|-nan' "$TMP/TOOLARGE.log")"
# ...and the diagnostic says nothing about time step, CFL or stability.
if grep -qiE "traveled more than one bin.*(cfl|time.?step|stability)" "$TMP/TOOLARGE.log"; then
  echo "DIAGNOSTIC_MENTIONS_TIMESTEP=yes"
else
  echo "DIAGNOSTIC_MENTIONS_TIMESTEP=no"
fi
exit 0
