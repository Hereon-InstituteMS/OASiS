#!/bin/bash
# Tier-2 for fourc::particle_pd#1 — DOMAINBOUNDINGBOX really must enclose every
# particle, but the consequence of getting it wrong is NOT a crash.
#
# Claimed:  "the simulation crashes", with a runtime abort
#           `particle outside DOMAINBOUNDINGBOX at step N` /
#           `BinningStrategy: particle position out of bounds`, printing the
#           offending position.
# Observed: 4C silently DELETES the offending particles and carries on.  The
#           only trace is one line —
#             on processor 0 removed N particle(s) being outside the computational domain!
#           — with no id, no position, no step number and no phase.  It fires
#           once at setup for particles that start outside, and again later in
#           the run for particles that leave.  The simulation then runs to the
#           end with a mutilated body and the failure surfaces at the very end
#           as "expected 10 tests but performed 0", because the result-tested
#           particle ids no longer exist.
#
# On the upstream 2-D PD deck a 40 x 40 box is shrunk to 28 x 40 in x.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
BOX='DOMAINBOUNDINGBOX: "-20.0 -20.0 -0.01 20.0 20.0 0.01"'
grep -qF "$BOX" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/big.yaml"
sed 's|DOMAINBOUNDINGBOX: "-20.0 -20.0 -0.01 20.0 20.0 0.01"|DOMAINBOUNDINGBOX: "-20.0 -20.0 -0.01 8.0 20.0 0.01"|' "$BASE" > "$TMP/small.yaml"

probe BIGBOX   "$TMP/big.yaml"
probe SMALLBOX "$TMP/small.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BIGBOX.log"
echo "BIGBOX_REMOVALS=$(grep -c 'being outside the computational domain!' "$TMP/BIGBOX.log")"

# The real diagnostic, and how little it says.
grep -m1 -oE "on processor 0 removed [0-9]+ particle\(s\) being outside the computational domain!" "$TMP/SMALLBOX.log"
echo "SMALLBOX_REMOVAL_EVENTS=$(grep -c 'being outside the computational domain!' "$TMP/SMALLBOX.log")"
# ...more than one event, i.e. it also fires mid-run, not only at setup.
if [ "$(grep -c 'being outside the computational domain!' "$TMP/SMALLBOX.log")" -gt 1 ]; then
  echo "REMOVAL_ALSO_HAPPENS_MID_RUN=yes"
else
  echo "REMOVAL_ALSO_HAPPENS_MID_RUN=no"
fi
# The run does not abort where the particle leaves; it dies at the result test.
grep -m1 -F "expected 10 tests but performed 0" "$TMP/SMALLBOX.log"
grep -m1 -oE "4C_utils_result_test\.cpp, line [0-9]+" "$TMP/SMALLBOX.log"
# Neither claimed abort string exists anywhere.
echo "CLAIMED_PARTICLE_OUTSIDE_TEXT=$(grep -ci 'particle outside DOMAINBOUNDINGBOX' "$TMP/SMALLBOX.log")"
echo "CLAIMED_BINNINGSTRATEGY_OUT_OF_BOUNDS_TEXT=$(grep -ci 'position out of bounds' "$TMP/SMALLBOX.log")"
exit 0
