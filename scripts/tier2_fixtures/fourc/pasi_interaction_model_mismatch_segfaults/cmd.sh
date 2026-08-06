#!/bin/bash
# Tier-2 for fourc::pasi#1 — the particle interaction model must match the
# materials and the wall treatment, and 4C does not check it.
#
# Claimed:  a mismatch "produces 'interaction type mismatch' or silent zero-force
#           coupling".
# Observed: neither.  Switching INTERACTION from DEM to SPH on the upstream
#           two-way DEM/membrane PASI deck — leaving MAT_ParticleDEM,
#           MAT_ParticleWallDEM and the PARTICLE DYNAMIC/DEM section exactly
#           where they were, which is precisely the mismatch the entry describes
#           — makes 4C print its time-stepping banner and then take
#             Signal: Segmentation fault (11) / Address not mapped (1)
#           with no PROC 0 ERROR block, no message, no MPI_Abort banner and a
#           signal exit status rather than 1.  There is no 'interaction type
#           mismatch' string in the output and the coupling never gets far
#           enough to be silently zero.
#
# That matters operationally: an agent watching for a non-zero exit plus a
# diagnostic gets a bare crash, and one watching for zero forces gets no output
# to inspect at all.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream pasi_twoway_norelax_particle_dem_1d_normalcontact_linspring_walldiscretcond.4C.yaml) || exit 3
grep -q '  INTERACTION: "DEM"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q 'MAT_ParticleDEM'      "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/dem.yaml"
sed 's|  INTERACTION: "DEM"|  INTERACTION: "SPH"|' "$BASE" > "$TMP/mismatch.yaml"
# The DEM materials and the DEM sub-section stay exactly as they were.
echo "DEM_MATERIALS_STILL_PRESENT=$(grep -c 'MAT_ParticleDEM\|MAT_ParticleWallDEM' "$TMP/mismatch.yaml")"
echo "DEM_SUBSECTION_STILL_PRESENT=$(grep -c '^PARTICLE DYNAMIC/DEM:' "$TMP/mismatch.yaml")"

probe DEM "$TMP/dem.yaml"
grep -m1 -F "processor 0 finished normally" "$TMP/DEM.log"
echo "DEM_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/DEM.log")"

# Keep the raw status: this arm dies on a signal, not with exit 1.
run4c "$TMP/mismatch.yaml" "$TMP/o_MISMATCH" > "$TMP/MISMATCH.log" 2>&1
MISMATCH_STATUS=$?
echo "EXIT_MISMATCH=$MISMATCH_STATUS"

grep -m1 -F "Overview of chosen time stepping" "$TMP/MISMATCH.log"
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/MISMATCH.log"
grep -m1 -F "Signal code: Address not mapped (1)" "$TMP/MISMATCH.log"
echo "MISMATCH_PROC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/MISMATCH.log")"
echo "CLAIMED_INTERACTION_TYPE_MISMATCH_TEXT=$(grep -ci 'interaction type mismatch' "$TMP/MISMATCH.log")"
echo "MISMATCH_REACHED_RESULT_TEST=$(grep -c 'is WRONG --> actresult=\|is CORRECT, abs' "$TMP/MISMATCH.log")"
if [ "$MISMATCH_STATUS" -gt 128 ]; then
  echo "MISMATCH_DIED_ON_A_SIGNAL=yes"
else
  echo "MISMATCH_DIED_ON_A_SIGNAL=no"
fi
exit 0
