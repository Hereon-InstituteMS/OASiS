#!/bin/bash
# Tier-2 for fourc::particles#13 — BIN_SIZE_LOWER_BOUND is checked against the
# INTERACTION DISTANCE the active method computes, which is a different number
# per method and is not anything you wrote.
#
# T2_MUTATE=1 removes every edit; the bin sizes stay adequate, nothing aborts
# and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

DEM=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
SPH=$(upstream particle_sph_1d_pressurewave_boundary_densitysummation_cubicspline_adami.4C.yaml) || exit 3
LN=$(upstream particle_dem_1d_radius_lognormal_distribution.4C.yaml) || exit 3

python3 - "$DEM" "$SPH" "$LN" "$TMP" "$MUTATE" <<'PY'
import sys
dem, sph, ln = (open(sys.argv[i]).read() for i in (1, 2, 3))
tmp, mutate = sys.argv[4], sys.argv[5] == "1"
def w(n, src, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
# DEM: the distance is 2 * MAX_RADIUS, so a bin below it is refused
w("demsmall", dem, dem.replace("  BIN_SIZE_LOWER_BOUND: 0.025", "  BIN_SIZE_LOWER_BOUND: 0.005"))
# SPH: the distance is the largest INITRADIUS, a different quantity entirely
w("sphsmall", sph, sph.replace("  BIN_SIZE_LOWER_BOUND: 0.012", "  BIN_SIZE_LOWER_BOUND: 0.005"))
# The trap: widening MAX_RADIUS trips a bin size that was previously fine,
# because a radius DISTRIBUTION sets the largest radius at run time.
w("wider", ln, ln.replace("  MAX_RADIUS: 0.012", "  MAX_RADIUS: 0.5")
                 .replace("  MIN_RADIUS: 0.008", "  MIN_RADIUS: 1e-06"))
PY

probe DEMSMALL "$TMP/demsmall.yaml"
probe SPHSMALL "$TMP/sphsmall.yaml"
probe WIDER    "$TMP/wider.yaml"
probe LNBASE   "$LN"

# The message prints BOTH numbers, which is what makes it directly actionable.
grep -m1 -oE "the particle interaction distance is larger than the minimal bin size \(.*\)!" "$TMP/DEMSMALL.log"
echo "DEM_BIN_REFUSED=$(grep -c 'particle interaction distance is larger than the minimal bin size' "$TMP/DEMSMALL.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# And the finding that falsified the drafted claim: the SAME edit on the SPH
# deck -- BIN_SIZE_LOWER_BOUND cut far below the support radius -- does NOT
# fire the check, because the key is only a LOWER BOUND and the engine sizes
# bins to divide the domain. Same key, same direction, opposite outcome.
echo "SPH_BIN_REFUSED=$(grep -c 'particle interaction distance is larger than the minimal bin size' "$TMP/SPHSMALL.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "SPH_SMALL_BIN_STILL_PASSES=$(grep -cE '^OK \(' "$TMP/SPHSMALL.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
grep -m1 -oE "4C_particle_interaction_base.cpp" "$TMP/DEMSMALL.log" | head -1 | sed 's/^/SOURCE_FILE=/'
echo "DEM_ABORTS_AT_SETUP=$(grep -c '^TIME:' "$TMP/DEMSMALL.log")"
# Bit-identity is the sharp version of "the SPH edit changed nothing": compare
# the verdict lines of the cut-bound run against the untouched deck.
probe SPHBASE "$SPH"
grep -E "is CORRECT|is WRONG" "$TMP/SPHSMALL.log" > "$TMP/v_small"
grep -E "is CORRECT|is WRONG" "$TMP/SPHBASE.log"  > "$TMP/v_base"
echo "SPH_VERDICT_LINES=$(wc -l < "$TMP/v_base")"
cmp -s "$TMP/v_small" "$TMP/v_base" && echo "SPH_CUT_BOUND_IS_BIT_IDENTICAL=yes" || echo "SPH_CUT_BOUND_IS_BIT_IDENTICAL=no"
# The trap: the deck's own bin size was fine until MAX_RADIUS moved.
grep -m1 -E "^OK \(" "$TMP/LNBASE.log" && echo "LNBASE_PASSES=yes"
echo "WIDENING_MAX_RADIUS_TRIPS_THE_SAME_CHECK=$(grep -c 'particle interaction distance is larger than the minimal bin size' "$TMP/WIDER.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "WIDER_TOUCHED_THE_BIN_SIZE=$( { grep -q 'BIN_SIZE_LOWER_BOUND: 0.025' "$TMP/wider.yaml"; } && echo no || echo yes)"
exit 0
