#!/bin/bash
# Tier-2 for fourc::particle_sph#2 — fluid that leaves DOMAINBOUNDINGBOX during
# the run is deleted, not diagnosed.
#
# Claimed:  "abort `particle outside DOMAINBOUNDINGBOX at step N`; in SPH this
#           typically triggers when a splash particle exits the bounding
#           y-extent on a dam-break case."
# Observed: no abort.  The bounding box here is left exactly as upstream wrote
#           it — the ONLY change is the direction and size of gravity, which
#           flings the free-surface column out through the far x face instead of
#           settling it — and 4C responds by quietly deleting particles as they
#           go, 48 separate times, each with the same contentless line:
#             on processor 0 removed 1 particle(s) being outside the computational domain!
#           No id, no position, no step, no phase, no abort.  The run reaches
#           the end and only then fails, with "expected 5 tests but performed 0"
#           because the result-tested particle is among the deleted.
#
# Because the geometry is untouched and the baseline reports zero removals,
# every one of those 48 deletions is mid-run by construction — which is the part
# of the claim ("throughout the simulation, including splashing and fluid
# expansion") that a startup-only check would miss.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_1d_hydrostatic_freesurface_densityintegration_cubicspline_adami.4C.yaml) || exit 3
grep -qF 'GRAVITY_ACCELERATION: "-0.0001 0.0 0.0"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -qF 'DOMAINBOUNDINGBOX: "-0.5 -0.15 -0.15 5.0 0.15 0.15"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/settles.yaml"
sed 's|GRAVITY_ACCELERATION: "-0.0001 0.0 0.0"|GRAVITY_ACCELERATION: "0.01 0.0 0.0"|' "$BASE" > "$TMP/flung.yaml"

# The bounding box is byte-identical in both arms; only the flow differs.
echo "BOX_IS_UNCHANGED=$( a=$(grep -c 'DOMAINBOUNDINGBOX: "-0.5 -0.15 -0.15 5.0 0.15 0.15"' "$TMP/settles.yaml"); \
  b=$(grep -c 'DOMAINBOUNDINGBOX: "-0.5 -0.15 -0.15 5.0 0.15 0.15"' "$TMP/flung.yaml"); \
  [ "$a" = "$b" ] && echo yes || echo no )"

probe SETTLES "$TMP/settles.yaml"
probe FLUNG   "$TMP/flung.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SETTLES.log"
echo "SETTLES_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SETTLES.log")"
echo "SETTLES_REMOVALS=$(grep -c 'being outside the computational domain!' "$TMP/SETTLES.log")"

grep -m1 -F "on processor 0 removed 1 particle(s) being outside the computational domain!" "$TMP/FLUNG.log"
echo "FLUNG_REMOVAL_EVENTS=$(grep -c 'being outside the computational domain!' "$TMP/FLUNG.log")"
# All of them are mid-run: the baseline, same geometry, removes nothing.
if [ "$(grep -c 'being outside the computational domain!' "$TMP/SETTLES.log")" = "0" ] \
   && [ "$(grep -c 'being outside the computational domain!' "$TMP/FLUNG.log")" -gt 10 ]; then
  echo "REMOVALS_ARE_ALL_MID_RUN=yes"
else
  echo "REMOVALS_ARE_ALL_MID_RUN=no"
fi
grep -m1 -F "expected 5 tests but performed 0" "$TMP/FLUNG.log"
echo "CLAIMED_PARTICLE_OUTSIDE_TEXT=$(grep -ci 'particle outside DOMAINBOUNDINGBOX' "$TMP/FLUNG.log")"
echo "FLUNG_PROC_ERROR_BLOCKS_BEFORE_RESULT_TEST=$(grep -c 'PROC 0 ERROR' "$TMP/FLUNG.log")"
exit 0
