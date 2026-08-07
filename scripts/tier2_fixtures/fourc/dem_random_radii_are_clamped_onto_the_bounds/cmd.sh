#!/bin/bash
# Tier-2 for fourc::particle_dem#5 — the random radius options clamp, they do
# not redraw.
#
# Claimed: NormalRadiusDistribution / LogNormalRadiusDistribution draw once per
#          particle and clamp the draw into [MIN_RADIUS, MAX_RADIUS], so the
#          tails pile up exactly on the bounds and the realised distribution is
#          the truncated one. 4C says nothing about clamping.
#
# Measured directly rather than by proxy: the radii are read out of the result
# verdicts and compared bit-exactly against MIN_RADIUS and MAX_RADIUS.
#
# T2_MUTATE=1 removes the pathology by widening the bounds far outside the
# draw, so nothing is clamped, CLAMPED_ONTO_A_BOUND drops to 0 and the fixture
# must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_dem_1d_radius_lognormal_distribution.4C.yaml) || exit 3
grep -q "RADIUSDISTRIBUTION_SIGMA: 0.4" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "RANDSEED: 1" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_lost_its_seed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
wide = src.replace("  MIN_RADIUS: 0.008", "  MIN_RADIUS: 1e-06").replace(
                   "  MAX_RADIUS: 0.012", "  MAX_RADIUS: 1.0")
open(tmp + "/draw.yaml", "w").write(wide if mutate else src)
# no seed: the same deck draws a different realisation
open(tmp + "/noseed.yaml", "w").write(src.replace("  RANDSEED: 1\n", ""))
# a distribution with no sigma at all
open(tmp + "/nosigma.yaml", "w").write(src.replace("  RADIUSDISTRIBUTION_SIGMA: 0.4\n", ""))
PY

probe DRAW    "$TMP/draw.yaml"
probe NOSEED  "$TMP/noseed.yaml"
probe NOSIGMA "$TMP/nosigma.yaml"

# Count how many radii sit bit-exactly on a bound. The upstream deck's own
# RESULT DESCRIPTION carries the drawn radii, so read them from the verdict
# lines instead of trusting the expected values in the file.
python3 - "$TMP/DRAW.log" "$TMP/draw.yaml" <<'PY'
import re, sys
log, deck = open(sys.argv[1]).read(), open(sys.argv[2]).read()
lo = float(re.search(r"MIN_RADIUS:\s*(\S+)", deck).group(1))
hi = float(re.search(r"MAX_RADIUS:\s*(\S+)", deck).group(1))
# CORRECT lines carry no number, so take the expected value out of the deck for
# those and the actual value out of the verdict for the WRONG ones.
vals = [float(v) for v in re.findall(r'QUANTITY: "radius"\n\s+VALUE: (\S+)', deck)]
act  = [float(m) for m in re.findall(r"radius\s+is WRONG --> actresult=\s*(\S+)", log)]
radii = act if act else vals
print("RADII_READ=%d" % len(radii))
print("CLAMPED_ONTO_A_BOUND=%d" % sum(1 for r in radii if r == lo or r == hi))
print("ALL_STRICTLY_INSIDE=%s" % ("yes" if all(lo < r < hi for r in radii) else "no"))
PY

# 4C never says a word about clamping or truncation.
echo "CLAMP_WARNINGS=$(grep -ciE 'clamp|truncat|out of range|bound' "$TMP/DRAW.log")"
# Without RANDSEED the same deck draws something else.
echo "NOSEED_CHANGES_THE_DRAW=$(grep -c 'radius.*is WRONG' "$TMP/NOSEED.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# A distribution with no sigma is the one case that IS refused.
grep -m1 -F "RADIUSDISTRIBUTION_SIGMA is not set but required for a radius distribution." "$TMP/NOSIGMA.log"   && echo "NOSIGMA_ABORTS=yes" || echo "NOSIGMA_ABORTS=no"
exit 0
