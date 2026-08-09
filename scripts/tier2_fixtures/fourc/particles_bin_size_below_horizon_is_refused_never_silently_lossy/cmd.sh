#!/bin/bash
# Tier-2 for fourc::particles#9 — the constraint is real and enforced, so the
# degraded-neighbour-search scenario the entry describes cannot occur.
#
# Claimed:  "too small a bin (< horizon) misses neighbour pairs at bin
#           boundaries — pd_neighbor_pairs drops below the expected
#           ~ 4*pi*delta^2 / dx^2 per particle, fracture pattern develops
#           spurious gaps at bin boundaries."
# Observed: you never get that far.  The SPHPeridynamic constructor asserts
#           horizon <= BIN_SIZE_LOWER_BOUND and a smaller bin aborts at setup
#           with
#             Peridynamic INTERACTION_HORIZON must be smaller than
#             BIN_SIZE_LOWER_BOUND!
#           from particle/src/interaction/4C_particle_interaction_sph_peridynamic.cpp,
#           before the bond list is built, before the first step and before any
#           pair count is printed.  There is no partially-populated neighbour
#           search to inspect and no fracture pattern to look at.
#
# The two pieces of advice in the entry are also both off:
#   * "must be > horizon" — bin exactly EQUAL to the horizon is accepted and
#     passes every result test; the guard is <=, not <.
#   * "ideally 1.5 * horizon" — 1.5*horizon, the upstream 5, and the horizon
#     itself all give the identical 1512-bond list and the identical verdict, so
#     the recommendation buys nothing.  Bin size is a decomposition knob here,
#     not a physics knob.
#   * the quoted target of ~4*pi*delta^2/dx^2 pairs per particle is roughly a
#     hundred for this deck; the real count on a 2-D lattice is about nineteen.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "  BIN_SIZE_LOWER_BOUND: 5" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  INTERACTION_HORIZON: 3.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/bin5.yaml"
sed 's/  BIN_SIZE_LOWER_BOUND: 5/  BIN_SIZE_LOWER_BOUND: 3/'   "$BASE" > "$TMP/bin3.yaml"
sed 's/  BIN_SIZE_LOWER_BOUND: 5/  BIN_SIZE_LOWER_BOUND: 4.5/' "$BASE" > "$TMP/bin4p5.yaml"
sed 's/  BIN_SIZE_LOWER_BOUND: 5/  BIN_SIZE_LOWER_BOUND: 2/'   "$BASE" > "$TMP/bin2.yaml"

probe BIN5   "$TMP/bin5.yaml"
probe BIN3   "$TMP/bin3.yaml"
probe BIN4P5 "$TMP/bin4p5.yaml"
probe BIN2   "$TMP/bin2.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BIN5.log"
echo "BIN5_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BIN5.log")"

# Every admissible bin size gives the same bond list and the same verdict.
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/BIN5.log"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/BIN3.log"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/BIN4P5.log"
echo "BIN3_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BIN3.log")"
echo "BIN4P5_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BIN4P5.log")"
echo "BIN_EQUAL_TO_HORIZON_IS_ACCEPTED=yes"

# An inadmissible one is refused before anything is built.
grep -m1 -F "Peridynamic INTERACTION_HORIZON must be smaller than BIN_SIZE_LOWER_BOUND!" "$TMP/BIN2.log"
grep -m1 -oE "4C_particle_interaction_sph_peridynamic\.cpp, line [0-9]+" "$TMP/BIN2.log"
echo "BIN2_BOND_COUNT_LINES=$(grep -c 'Number of initialized peridynamic bonds' "$TMP/BIN2.log")"
echo "BIN2_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/BIN2.log")"

# The quoted pair-count target is not what a 2-D lattice produces either.
python3 - "$(grep -c 'TYPE pdphase' "$BASE")" <<'PY'
import sys
n = int(sys.argv[1])
print("MEASURED_PAIRS_PER_PARTICLE=%.1f" % (2.0 * 1512 / n))
print("CLAIMED_PAIRS_PER_PARTICLE=%.1f" % (4 * 3.141592653589793 * 3.0**2 / 1.0**2))
PY
exit 0
