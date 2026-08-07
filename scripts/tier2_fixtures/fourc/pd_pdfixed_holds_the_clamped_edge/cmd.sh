#!/bin/bash
# Tier-2 for fourc::particle_pd#11 — PDFIXED 1 on a particle line really does pin
# that particle, and dropping it really does let the specimen move off as a body.
#
# NOTE ON PROVENANCE.  PDFIXED and the deck used here,
# particle_sph_2d_pdbody_precrack_fixedflag.4C.yaml, are NOT in upstream 4C
# main.  They come from local branch work on bond-based peridynamics in the
# checkout this was run against.  The fixture aborts loudly rather than passing
# silently where the deck is absent.
#
# The deck clamps the whole left column (ten particles at x = -4.5) with
# PDFIXED 1 and pulls the plate with gravity in +x.  Particle 0 is result-tested
# at exactly its reference position (-4.5, -4.5) with zero velocity.
#
# Strip ' PDFIXED 1' from the ten particle lines and nothing else changes — the
# same pre-crack, the same 974 initialised bonds, the same damage at the probe
# particle.  4C prints no warning.  But the clamp is gone, so the plate is no
# longer held: ten of the fourteen result tests fail, including the clamped
# particle's own position and velocity, while the two pd_damage_phi tests still
# pass — a reader checking only the crack sees a healthy run.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_precrack_fixedflag.4C.yaml) || exit 3
grep -q 'PDFIXED 1"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
echo "CLAMPED_PARTICLE_LINES=$(grep -c 'PDFIXED 1"' "$BASE")"

cp "$BASE" "$TMP/clamped.yaml"
sed 's/ PDFIXED 1"/"/' "$BASE" > "$TMP/free.yaml"

probe CLAMPED "$TMP/clamped.yaml"
probe FREE    "$TMP/free.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/CLAMPED.log"
echo "CLAMPED_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/CLAMPED.log")"
# The clamped particle sits on its reference position to the last bit.
echo "CLAMPED_EXACT_ZERO_DIFFS=$(grep -c 'is CORRECT, abs(diff)= 0.00000000000000000e+00' "$TMP/CLAMPED.log")"

# Removing the flag changes nothing structural — same crack, same bonds.
grep -m1 -F "Number of pre-crack segments: 1" "$TMP/FREE.log"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 974" "$TMP/FREE.log"
echo "FREE_WARNINGS=$(grep -ciE 'pdfixed|fixed particle|clamp' "$TMP/FREE.log")"
# ...but the specimen is no longer held.
echo "FREE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/FREE.log")"
grep -m1 -F "Result check failed with 10 errors out of 14 tests" "$TMP/FREE.log"
# The damage tests still pass, so a crack-only check misses it entirely.
echo "FREE_DAMAGE_TESTS_STILL_PASSING=$(grep -c 'pd_damage_phi.*is CORRECT' "$TMP/FREE.log")"
exit 0
