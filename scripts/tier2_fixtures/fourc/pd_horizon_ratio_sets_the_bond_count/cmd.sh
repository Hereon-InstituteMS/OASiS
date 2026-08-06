#!/bin/bash
# Tier-2 for fourc::particle_pd#3 — the horizon ratio m = INTERACTION_HORIZON /
# PERIDYNAMIC_GRID_SPACING is a first-class discretisation parameter, and 4C
# gives you a direct read-out of it that the entry did not mention.
#
# Claimed:  m = 2 puts the stress field 10-20% off the analytic elasticity
#           solution, m = 3 gives 1-3%, m = 4 gives < 1%, with m = 3-4 the
#           practical sweet spot.
# Observed: none of that is reachable from a 4C log, and the upstream PD deck
#           does not show a plateau at m = 3-4 — the answer keeps drifting
#           monotonically all the way to m = 5.  What IS directly observable is
#           the line 4C prints once per run,
#             Number of initialized peridynamic bonds on this proc: N
#           which is the discrete count of the horizon neighbourhood and grows
#           like m^2 in 2-D.  On this deck (dx = 1) it goes
#             m=1.5 -> 519,  m=2 -> 753,  m=3 -> 1512,  m=4 -> 2260,  m=5 -> 3155.
#           Halving m from 3 to 2 halves the bond count, which is the concrete
#           thing "m must be at least 3" is protecting you from.
#
# BIN_SIZE_LOWER_BOUND is raised to 8 in every arm so that the m = 4 and m = 5
# arms clear 4C's horizon <= bin-size assertion; that raise alone changes
# nothing (the m = 3 arm still passes the deck's own result tests), which the
# fixture also checks.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "INTERACTION_HORIZON: 3.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "PERIDYNAMIC_GRID_SPACING: 1.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "BIN_SIZE_LOWER_BOUND: 5" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

arm() {  # $1 = horizon, $2 = label
  sed -e "s|INTERACTION_HORIZON: 3.0|INTERACTION_HORIZON: $1|" \
      -e 's|BIN_SIZE_LOWER_BOUND: 5|BIN_SIZE_LOWER_BOUND: 8|' "$BASE" > "$TMP/$2.yaml"
  probe "$2" "$TMP/$2.yaml"
  echo "BONDS_$2=$(grep -m1 -oE 'Number of initialized peridynamic bonds on this proc: [0-9]+' "$TMP/$2.log" | grep -oE '[0-9]+$')"
}

arm 2.0 M2
arm 3.0 M3
arm 4.0 M4
arm 5.0 M5

# m = 3 is the upstream setting: bin-size raise alone leaves the deck passing.
grep -m1 -F "processor 0 finished normally" "$TMP/M3.log"
echo "M3_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/M3.log")"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/M3.log"

# Halving the horizon halves the neighbourhood and moves the answer.
echo "M2_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/M2.log")"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 753" "$TMP/M2.log"

# And there is no plateau: m = 4 and m = 5 keep moving away from m = 3.
echo "M4_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/M4.log")"
echo "M5_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/M5.log")"
python3 - "$TMP/M2.log" "$TMP/M4.log" "$TMP/M5.log" <<'PY'
import re, sys
def posx(p):
    for line in open(p):
        if "posx" in line and "actresult=" in line:
            return float(re.search(r"actresult=\s*(-?[0-9.]+e[+-][0-9]+)", line).group(1))
    return None
ref = 7.92632290305826448   # the deck's own m = 3 reference for particle 132
d = [abs(posx(p) - ref) for p in sys.argv[1:]]
print("DIST_FROM_M3_M2=%.3e" % d[0])
print("DIST_FROM_M3_M4=%.3e" % d[1])
print("DIST_FROM_M3_M5=%.3e" % d[2])
print("HORIZON_ANSWER_PLATEAUS_ABOVE_M3=%s" % ("no" if d[2] > d[1] > 0 else "yes"))
PY
exit 0
