#!/bin/bash
# Tier-2 for fourc::particles#0 — the rule holds, the Signal is fabricated, and
# the prescribed three-key SPH block really is enough.
#
# Claimed:  "omitting the SPH section gives pd_neighbor_pairs = 0 at runtime
#           (visible in stderr) and zero displacement, with NO error message —
#           4C happily runs a no-force simulation."
# Observed: it does not run at all.  Deleting PARTICLE DYNAMIC/SPH from upstream
#           particle_sph_2d_pdbody_gravity.4C.yaml aborts before the first time
#           step with
#             negative initial particle spacing!
#           from particle/src/interaction/4C_particle_interaction_sph.cpp —
#           INITIALPARTICLESPACING lives in that sub-section and sets every
#           particle's mass, so its default of -1 is caught immediately.  Not one
#           time step is taken, so there is no displacement to look at.
#
# The quoted symptom is worse than merely absent: a HEALTHY run prints
# "Number of pd_neighbor_pairs in peridynamic evaluation on this proc: 0" for
# most of its steps (that counter is the body-to-body CONTACT pair list, not the
# bond list), so an agent taught to read a zero there as a broken deck will
# condemn a correct one.
#
# The entry's remedy is checked too: cutting the SPH block down to exactly the
# three keys it names — KERNEL, KERNEL_SPACE_DIM, INITIALPARTICLESPACING —
# reproduces the upstream run bit for bit, so BOUNDARYPARTICLEFORMULATION and
# TRANSPORTVELOCITYFORMULATION are inert for a pure PD body.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3

python3 - "$BASE" "$TMP/full.yaml" "$TMP/minsph.yaml" "$TMP/nosph.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = '''PARTICLE DYNAMIC/SPH:
  KERNEL: QuinticSpline
  KERNEL_SPACE_DIM: Kernel2D
  INITIALPARTICLESPACING: 1.0
  BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
  TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity
'''
assert old in t, "upstream deck no longer carries the five-key SPH sub-section"
three = '''PARTICLE DYNAMIC/SPH:
  KERNEL: QuinticSpline
  KERNEL_SPACE_DIM: Kernel2D
  INITIALPARTICLESPACING: 1.0
'''
open(sys.argv[2], "w").write(t)
open(sys.argv[3], "w").write(t.replace(old, three))
open(sys.argv[4], "w").write(t.replace(old, ""))
print("SPH_KEYS_FULL=%d"   % (old.count("\n") - 1))
print("SPH_KEYS_MINIMAL=%d" % (three.count("\n") - 1))
PY

probe FULL   "$TMP/full.yaml"
probe MINSPH "$TMP/minsph.yaml"
probe NOSPH  "$TMP/nosph.yaml"

# Baseline.
grep -m1 -F "processor 0 finished normally" "$TMP/FULL.log"
echo "FULL_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/FULL.log")"
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/FULL.log"

# The remedy the entry prescribes is exactly sufficient: same bonds, same answer
# to the last bit.
grep -m1 -F "Number of initialized peridynamic bonds on this proc: 1512" "$TMP/MINSPH.log"
echo "MINSPH_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/MINSPH.log")"
echo "MINSPH_EXACT_ZERO_DIFFS=$(grep -c 'is CORRECT, abs(diff)= 0.00000000000000000e+00' "$TMP/MINSPH.log")"

# Deleting the sub-section is a hard abort, not a silent no-force run.
grep -m1 -F "negative initial particle spacing!" "$TMP/NOSPH.log"
grep -m1 -oE "4C_particle_interaction_sph\.cpp, line [0-9]+" "$TMP/NOSPH.log"
echo "NOSPH_TIME_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/NOSPH.log")"
echo "CLAIMED_PD_NEIGHBOR_PAIRS_EQUALS_ZERO_TEXT=$(grep -ci 'pd_neighbor_pairs = 0' "$TMP/NOSPH.log")"

# ...and a zero pair count is the normal state of a correct run, so the quoted
# symptom would condemn the baseline.
echo "FULL_STEPS_WITH_ZERO_PD_NEIGHBOR_PAIRS=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation on this proc: 0$' "$TMP/FULL.log")"
exit 0
