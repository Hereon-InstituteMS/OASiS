#!/bin/bash
# Tier-2 for fourc::particle_pd#8 — the physics is right, the Signal describes
# something 4C makes impossible.
#
# Claimed:  "a bond-based PD problem in the PARTICLE DYNAMIC/PD MAT_ParticlePD
#           MATERIALS entry at nu != these fixed values RUNS but the resulting
#           Poisson contraction matches 0.25 (2D) or 0.33 (3D) regardless of the
#           input nu."
# Observed: there is no input nu.  MAT_ParticlePD takes exactly four keys —
#           INITRADIUS, INITDENSITY, YOUNG, CRITICAL_STRETCH — and adding a
#           Poisson ratio under any of the usual spellings is a hard parse
#           error: "Could not match this input" from
#           global_data/4C_global_data_read.cpp, with 4C echoing the whole
#           MATERIALS block back.  So the deck never runs and there is no
#           contraction to compare.
#
# The underlying restriction is real — bond-based PD fixes nu at 1/4 in plane
# strain and 1/3 in 3-D — but 4C enforces it by not offering the knob, not by
# silently overriding one.  An agent told to "check whether the contraction
# matches 0.25 despite your nu" will be looking for an observation that cannot
# be made.  Three spellings are tried so the fixture does not rest on one guess.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
MATBLK='  - MAT: 2
    MAT_ParticlePD:
      INITRADIUS: 0.5'
grep -qF "$MATBLK" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/asis.yaml"
for key in NUE POISSONRATIO NU; do
  python3 - "$BASE" "$TMP/$key.yaml" "$key" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = "      YOUNG: 190.0e3\n      CRITICAL_STRETCH: 295262.0"
assert old in t
new = "      YOUNG: 190.0e3\n      %s: 0.3\n      CRITICAL_STRETCH: 295262.0" % sys.argv[3]
open(sys.argv[2], "w").write(t.replace(old, new))
PY
done

probe ASIS "$TMP/asis.yaml"
probe NUE          "$TMP/NUE.yaml"
probe POISSONRATIO "$TMP/POISSONRATIO.yaml"
probe NU           "$TMP/NU.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/ASIS.log"
echo "ASIS_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/ASIS.log")"

grep -m1 -F "Could not match this input" "$TMP/NUE.log"
grep -m1 -oE "4C_global_data_read\.cpp, line [0-9]+" "$TMP/NUE.log"
# 4C echoes the offending block, so the reader at least sees what it rejected.
grep -m1 -F "MAT_ParticlePD:" "$TMP/NUE.log"

# Every spelling is rejected: there is no Poisson ratio to give.
echo "REJECTED_SPELLINGS=$( n=0; for k in NUE POISSONRATIO NU; do
  grep -q 'Could not match this input' "$TMP/$k.log" && n=$((n+1)); done; echo $n )"
# The deck therefore never reaches a time step, let alone a contraction.
echo "NUE_TIME_STEPS_RUN=$(grep -c 'pd_neighbor_pairs in peridynamic evaluation' "$TMP/NUE.log")"
exit 0
