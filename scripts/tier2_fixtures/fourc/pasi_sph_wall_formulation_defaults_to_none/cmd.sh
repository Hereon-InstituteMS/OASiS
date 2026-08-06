#!/bin/bash
# Tier-2 for fourc::pasi#5 — SPH-structure coupling needs an explicit wall
# treatment, and 4C's default is to have none.
#
# The knob is WALLFORMULATION in PARTICLE DYNAMIC/SPH.  Its default is
# NoWallFormulation, so leaving it out is not "using the plain coupling" — it is
# switching the boundary treatment off.  Upstream's piston deck sets
# VirtualParticleWallFormulation; deleting that one line is the whole mutation.
#
# The result is exactly the mechanism the entry describes, and it is silent:
# 4C emits no warning, runs the full coupled loop and reaches the result test,
# but the fluid particle's density falls from 9.99658304322867730e-01 to
# 8.99761790578682885e-01 — a 10% deficit, which is what a kernel whose support
# crosses the wall into empty space computes — and its z-velocity flips sign
# from -2.47255596595578497e-01 to 1.36328793070252097e+00, i.e. it is no longer
# being held off the piston face.  Six of the deck's eighteen result tests fail.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream pasi_twoway_norelax_particle_sph_3d_piston_virtualwall_densitysummation.4C.yaml) || exit 3
grep -q 'WALLFORMULATION: "VirtualParticleWallFormulation"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/virtualwall.yaml"
grep -v 'WALLFORMULATION' "$BASE" > "$TMP/nowall.yaml"
echo "WALLFORMULATION_LINES_REMOVED=$(( $(grep -c WALLFORMULATION "$TMP/virtualwall.yaml") - $(grep -c WALLFORMULATION "$TMP/nowall.yaml") ))"

probe VIRTUALWALL "$TMP/virtualwall.yaml"
probe NOWALL      "$TMP/nowall.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/VIRTUALWALL.log"
echo "VIRTUALWALL_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/VIRTUALWALL.log")"

# Switching the boundary treatment off is accepted without a word.
echo "NOWALL_WARNINGS=$(grep -ciE 'wall formulation|wallformulation|boundary treatment' "$TMP/NOWALL.log")"
echo "NOWALL_REACHED_RESULT_TEST=$(grep -c 'is WRONG --> actresult=\|is CORRECT, abs' "$TMP/NOWALL.log")"
echo "NOWALL_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NOWALL.log")"

# Density deficit: the kernel support reaches through the wall into nothing.
grep -m1 -F "is WRONG --> actresult= 8.99761790578682885e-01" "$TMP/NOWALL.log"
# And the particle is no longer held off the piston face — velz changes sign.
grep -m1 -F "is WRONG --> actresult= 1.36328793070252097e+00" "$TMP/NOWALL.log"

python3 - "$TMP/NOWALL.log" <<'PY'
import re, sys
for l in open(sys.argv[1]):
    if "density" in l and "actresult=" in l:
        v = float(re.search(r"actresult=\s*(-?[0-9.]+e[+-][0-9]+)", l).group(1))
        print("NOWALL_DENSITY_DEFICIT_PERCENT=%.1f" % ((1.0 - v) * 100))
        print("SPH_PARTICLES_SEE_VACUUM_ACROSS_THE_WALL=%s" % ("yes" if v < 0.95 else "no"))
        break
PY
exit 0
