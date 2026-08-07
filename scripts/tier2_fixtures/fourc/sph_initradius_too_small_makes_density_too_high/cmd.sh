#!/bin/bash
# Tier-2 for fourc::particle_sph#1 — the rule is right and the Signal points the
# wrong way.
#
# Claimed:  INITRADIUS is the kernel SUPPORT radius, not half the spacing; if
#           you set it to dx/2 "the density field is uniformly low
#           (~0.5 * INITDENSITY) because most particles are outside each others'
#           kernels".
# Observed: too-small a support makes the summation density come out uniformly
#           HIGH, not low.  On 4C's own 1-D density-summation deck (dx = 0.004,
#           INITRADIUS = 0.008 = 2*dx, the CubicSpline support), dropping
#           INITRADIUS to dx/2 = 0.002 gives density = 2.66666666666666652e+00 at
#           BOTH tested particles — 8/3, exactly the isolated-particle
#           self-density m*W(0) with m = rho*dx and W(0) = 2/(3h).  With no
#           neighbours inside the support each particle sees only itself, and
#           because the 1-D kernel carries a 1/h prefactor, shrinking h RAISES
#           the sum.  Anyone looking for ~0.5*INITDENSITY is looking at the
#           wrong end of the scale by a factor of five.
#
# The support-radius convention itself is confirmed against the upstream decks:
# CubicSpline decks use INITRADIUS = 2*dx, QuinticSpline decks use 3*dx.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
grep -q "INITIALPARTICLESPACING: 0.004" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/support.yaml"
python3 - "$BASE" "$TMP/halfspacing.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = """  - MAT: 1
    MAT_ParticleSPHFluid:
      INITRADIUS: 0.008"""
assert old in t, "upstream deck no longer sets the fluid INITRADIUS to 2*dx"
open(sys.argv[2], "w").write(t.replace(old, old.replace("INITRADIUS: 0.008", "INITRADIUS: 0.002")))
PY

# The upstream convention itself: 2*dx for CubicSpline, 3*dx for QuinticSpline.
QUINTIC=$(upstream particle_sph_2d_pdbody_interaction_soft.4C.yaml) || exit 3
echo "CUBICSPLINE_SUPPORT_IS_TWO_DX=$([ "$(grep -c 'INITRADIUS: 0.008' "$BASE")" -ge 1 ] && echo yes || echo no)"
echo "QUINTICSPLINE_SUPPORT_IS_THREE_DX=$([ "$(grep -c 'INITRADIUS: 0.3' "$QUINTIC")" -ge 1 ] && [ "$(grep -c 'INITIALPARTICLESPACING: 0.1' "$QUINTIC")" -ge 1 ] && echo yes || echo no)"

probe SUPPORTRADIUS "$TMP/support.yaml"
probe HALFSPACING   "$TMP/halfspacing.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SUPPORTRADIUS.log"
echo "SUPPORTRADIUS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SUPPORTRADIUS.log")"

echo "HALFSPACING_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/HALFSPACING.log")"
grep -m1 -F "is WRONG --> actresult= 2.66666666666666652e+00" "$TMP/HALFSPACING.log"
echo "HALFSPACING_WARNINGS=$(grep -ciE 'initradius|support radius|too few neighbou?r' "$TMP/HALFSPACING.log")"

python3 - "$TMP/HALFSPACING.log" <<'PY'
import re, sys
vals = [float(re.search(r"actresult=\s*(-?[0-9.]+e[+-][0-9]+)", l).group(1))
        for l in open(sys.argv[1]) if "density" in l and "actresult=" in l]
print("HALFSPACING_DENSITIES=%s" % ",".join("%.6f" % v for v in vals))
print("DENSITY_WENT_LOW_AS_CLAIMED=%s" % ("yes" if vals and max(vals) < 1.0 else "no"))
print("DENSITY_WENT_HIGH_INSTEAD=%s" % ("yes" if vals and min(vals) > 1.0 else "no"))
PY
exit 0
