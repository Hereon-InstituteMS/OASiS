#!/bin/bash
# Tier-2 for fourc::particle_pd#0 — the PARTICLE DYNAMIC/SPH sub-section really is
# mandatory for a peridynamic run, but BOTH of the diagnostics the entry quoted
# are wrong about it.
#
# Claimed:  removing the SPH parameters "causes the code to crash with
#           'pd_neighbor_pairs = 0'", or prints
#           'BOUNDARYPARTICLEFORMULATION not set' during SPH section parsing.
# Observed: it aborts in ParticleInteractionSPH::set_initial_states with
#           "negative initial particle spacing!" — INITIALPARTICLESPACING lives in
#           that sub-section and defaults to a negative sentinel, and the mass of
#           every particle is rho * INITIALPARTICLESPACING^dim.  Nothing about
#           neighbour pairs or boundary formulations is mentioned.
#
# The second half matters more.  4C does print a pd_neighbor_pairs line —
# "Number of pd_neighbor_pairs in peridynamic evaluation on this proc: N" — but
# that is the count of *inter-body contact* pairs, and it is legitimately 0 on
# every step of a healthy single-contact-free run.  The baseline deck, which
# passes all ten of its own result tests, prints it as 0 thousands of times.
# Treating "pd_neighbor_pairs = 0" as an error signal is therefore a false alarm
# generator, which is why the baseline count is asserted below.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
grep -q "INITIALPARTICLESPACING: 1.0" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/good.yaml"
python3 - "$BASE" "$TMP/bad.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = """PARTICLE DYNAMIC/SPH:
  KERNEL: QuinticSpline
  KERNEL_SPACE_DIM: Kernel2D
  INITIALPARTICLESPACING: 1.0
  BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
  TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity
"""
assert blk in t, "upstream deck no longer carries the PARTICLE DYNAMIC/SPH block"
open(sys.argv[2], "w").write(t.replace(blk, ""))
PY

probe WITHSPH "$TMP/good.yaml"
probe NOSPH   "$TMP/bad.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHSPH.log"
echo "BASELINE_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHSPH.log")"

# What 4C really says when the sub-section is gone.
grep -m1 -F "negative initial particle spacing!" "$TMP/NOSPH.log"
grep -m1 -oE "4C_particle_interaction_sph\.cpp, line [0-9]+" "$TMP/NOSPH.log"

# Neither claimed diagnostic exists.
echo "CLAIMED_PD_NEIGHBOR_PAIRS_EQUALS_TEXT=$(grep -c 'pd_neighbor_pairs = 0' "$TMP/NOSPH.log")"
echo "CLAIMED_BOUNDARYPARTICLEFORMULATION_NOT_SET=$(grep -ci 'BOUNDARYPARTICLEFORMULATION not set' "$TMP/NOSPH.log")"

# And the real pd_neighbor_pairs line reads 0 all over a run that PASSES.
grep -m1 -F "Number of pd_neighbor_pairs in peridynamic evaluation on this proc: 0" "$TMP/WITHSPH.log"
if [ "$(grep -c 'pd_neighbor_pairs in peridynamic evaluation on this proc: 0' "$TMP/WITHSPH.log")" -gt 100 ]; then
  echo "ZERO_PD_NEIGHBOR_PAIRS_IS_NORMAL_IN_A_PASSING_RUN=yes"
else
  echo "ZERO_PD_NEIGHBOR_PAIRS_IS_NORMAL_IN_A_PASSING_RUN=no"
fi
exit 0
