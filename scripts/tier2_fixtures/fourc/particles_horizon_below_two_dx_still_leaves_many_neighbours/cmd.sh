#!/bin/bash
# Tier-2 for fourc::particles#3 — the rule holds, the neighbour count does not.
#
# Claimed:  "setting INTERACTION_HORIZON < 2*dx in PARTICLE DYNAMIC/PD gives a
#           MAT_ParticlePD model with each particle only seeing 1-2 neighbours —
#           bond count is too sparse, stiffness is mesh-dependent and
#           convergence as dx -> 0 fails."
# Observed: on a square lattice the neighbourhood is never that thin.  Upstream
#           particle_sph_2d_pdbody_gravity.4C.yaml has 162 pdphase particles at
#           dx = 1.0.  At the entry's own m = 3 the bond list holds 1512 pairs.
#           Dropping to horizon 1.5 (m = 1.5, below the 2*dx line) still leaves
#           519 pairs, and even at horizon 1.0 — the tightest horizon that
#           connects anything at all — there are 279 pairs.  Per particle that is
#           roughly six and three partners, not one or two: the 3x3 stencil
#           minus the centre, then its four axial members.
#
# What the too-small horizon actually costs you is not silent mesh dependence:
# both under-resolved runs go unstable and 4C kills them with
#   a particle of phase 'pdphase' traveled more than one bin on this processor!
# after a few dozen steps, having printed no word about the horizon.  An agent
# told to diagnose this by counting neighbours will find plenty of them.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "  INTERACTION_HORIZON: 3.0" "$BASE"       || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  PERIDYNAMIC_GRID_SPACING: 1.0" "$BASE"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  BIN_SIZE_LOWER_BOUND: 5" "$BASE"        || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/m3.yaml"
sed 's/  INTERACTION_HORIZON: 3.0/  INTERACTION_HORIZON: 1.5/' "$BASE" > "$TMP/m1p5.yaml"
sed 's/  INTERACTION_HORIZON: 3.0/  INTERACTION_HORIZON: 1.0/' "$BASE" > "$TMP/m1.yaml"
echo "PDPHASE_PARTICLES=$(grep -c 'TYPE pdphase' "$BASE")"

probe M3   "$TMP/m3.yaml"
probe M1P5 "$TMP/m1p5.yaml"
probe M1   "$TMP/m1.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/M3.log"
echo "M3_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/M3.log")"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/M3.log"

# Below 2*dx the bond list thins out, but nowhere near to "1-2 neighbours".
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 519" "$TMP/M1P5.log"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 279" "$TMP/M1.log"
python3 - "$TMP/M1P5.log" "$TMP/M1.log" "$(grep -c 'TYPE pdphase' "$BASE")" <<'PY'
import re, sys
n = int(sys.argv[3])
for tag, path in (("M1P5", sys.argv[1]), ("M1", sys.argv[2])):
    b = int(re.search(r"Number of initialized peridynamic bonds on this proc: (\d+)",
                      open(path).read()).group(1))
    print("AVG_BONDS_PER_PARTICLE_%s=%.1f" % (tag, 2.0 * b / n))
PY
echo "CLAIMED_ONE_OR_TWO_NEIGHBOURS=no"

# The real symptom is an instability abort that never names the horizon.
grep -m1 -F "a particle of phase 'pdphase' traveled more than one bin on this processor!" "$TMP/M1P5.log"
grep -m1 -oE "4C_particle_algorithm\.cpp, line [0-9]+" "$TMP/M1P5.log"
echo "M1P5_HORIZON_WARNINGS=$(grep -ci 'horizon' "$TMP/M1P5.log")"
echo "M1P5_STEPS_BEFORE_ABORT=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/M1P5.log")"
exit 0
