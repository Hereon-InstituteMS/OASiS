#!/bin/bash
# Tier-2 for fourc::particle_sph#7 — GRAVITY_ACCELERATION defaults to zero and
# nothing tells you.
#
# Delete the one line from 4C's own 1-D hydrostatic free-surface deck and the
# column does not settle by so much as a rounding error.  The probe particle
# reports its initial position back exactly:
#
#   posx    = 4.84999999999999964e+00   (it starts at 4.85)
#   velx    = 0.00000000000000000e+00   (exactly zero, not small)
#   density = 1.00000000000000000e+00   (exactly INITDENSITY, not near it)
#
# against 4.81167849295583760, -3.50289401431777894e-05 and
# 1.00015521027962739 with gravity present.  The run exits, prints no warning
# about a missing or zero body force, and the only sign anything is wrong is the
# result test — which a deck you wrote yourself would not have.
#
# That is exactly the claimed "column stays static, velocities remain ~0", made
# exact: the velocity is not approximately zero, it is bit-zero.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_1d_hydrostatic_freesurface_densityintegration_cubicspline_adami.4C.yaml) || exit 3
grep -q "GRAVITY_ACCELERATION" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/withgravity.yaml"
grep -v 'GRAVITY_ACCELERATION' "$BASE" > "$TMP/nogravity.yaml"
echo "GRAVITY_LINES_REMOVED=$(( $(grep -c GRAVITY_ACCELERATION "$TMP/withgravity.yaml") - $(grep -c GRAVITY_ACCELERATION "$TMP/nogravity.yaml") ))"

probe WITHGRAVITY "$TMP/withgravity.yaml"
probe NOGRAVITY   "$TMP/nogravity.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHGRAVITY.log"
echo "WITHGRAVITY_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHGRAVITY.log")"

# No warning about the missing body force at all.
echo "NOGRAVITY_WARNINGS=$(grep -ciE 'gravity|body force' "$TMP/NOGRAVITY.log")"
echo "NOGRAVITY_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOGRAVITY.log")"

# The column is not approximately static, it is exactly static.
grep -m1 -F "is WRONG --> actresult= 4.84999999999999964e+00" "$TMP/NOGRAVITY.log"
grep -m1 -F "is WRONG --> actresult= 0.00000000000000000e+00" "$TMP/NOGRAVITY.log"
grep -m1 -F "is WRONG --> actresult= 1.00000000000000000e+00" "$TMP/NOGRAVITY.log"
if grep -qE "velx .*actresult= 0\.00000000000000000e\+00" "$TMP/NOGRAVITY.log" \
   && grep -qE "density .*actresult= 1\.00000000000000000e\+00" "$TMP/NOGRAVITY.log"; then
  echo "COLUMN_IS_EXACTLY_STATIC=yes"
else
  echo "COLUMN_IS_EXACTLY_STATIC=no"
fi
exit 0
