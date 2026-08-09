#!/bin/bash
# Tier-2 for fourc::particle_pd#4 — BIN_SIZE_LOWER_BOUND really must not be
# smaller than the peridynamic horizon, but 4C does not let you find out the
# way the entry described.
#
# Claimed:  a silent degradation — "neighbor_count diagnostics from
#           BinningStrategy show ~50-70% of expected bonds" and the wave speed
#           comes out roughly the square root of the true one.
# Observed: a hard abort in the SPHPeridynamic constructor, before a single time
#           step, with the message
#             Peridynamic INTERACTION_HORIZON must be smaller than BIN_SIZE_LOWER_BOUND!
#           from particle/src/interaction/4C_particle_interaction_sph_peridynamic.cpp.
#           There is nothing silent about it and there are no bonds to miscount:
#           the "Number of initialized peridynamic bonds" line, which is the
#           only bond-count diagnostic 4C has, is never reached.
#
# Worth knowing: the guard is horizon <= bin size, so equality is accepted even
# though the message says "smaller than".  The fixture pins that too.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "BIN_SIZE_LOWER_BOUND: 5" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "INTERACTION_HORIZON: 3.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/bin5.yaml"
sed 's|BIN_SIZE_LOWER_BOUND: 5|BIN_SIZE_LOWER_BOUND: 2.0|' "$BASE" > "$TMP/bin2.yaml"
sed 's|BIN_SIZE_LOWER_BOUND: 5|BIN_SIZE_LOWER_BOUND: 3.0|' "$BASE" > "$TMP/bin3.yaml"

probe BIN5 "$TMP/bin5.yaml"   # bin  > horizon: upstream, passes
probe BIN3 "$TMP/bin3.yaml"   # bin == horizon: accepted despite the wording
probe BIN2 "$TMP/bin2.yaml"   # bin  < horizon: hard abort

grep -m1 -F "processor 0 finished normally" "$TMP/BIN5.log"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/BIN5.log"

# The abort, and where it comes from.
grep -m1 -F "Peridynamic INTERACTION_HORIZON must be smaller than BIN_SIZE_LOWER_BOUND!" "$TMP/BIN2.log"
grep -m1 -oE "4C_particle_interaction_sph_peridynamic\.cpp, line [0-9]+" "$TMP/BIN2.log"
# It never gets as far as counting bonds, so there is no degraded count to read.
echo "BIN2_BOND_COUNT_LINES=$(grep -c 'Number of initialized peridynamic bonds' "$TMP/BIN2.log")"
echo "BIN2_TIME_STEPS_RUN=$(grep -c 'pd_neighbor_pairs in peridynamic evaluation' "$TMP/BIN2.log")"
# The claimed silent-degradation wording is nowhere.
echo "CLAIMED_NEIGHBOR_COUNT_DIAGNOSTIC=$(grep -ci 'neighbor_count' "$TMP/BIN2.log")"

# Equality is accepted, and gives the same bond count as the upstream setting.
echo "BIN_EQUAL_TO_HORIZON_IS_ACCEPTED=$([ "$(grep -c 'INTERACTION_HORIZON must be smaller' "$TMP/BIN3.log")" = 0 ] && echo yes || echo no)"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/BIN3.log"
exit 0
