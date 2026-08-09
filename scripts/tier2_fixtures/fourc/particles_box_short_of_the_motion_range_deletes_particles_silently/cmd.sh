#!/bin/bash
# Tier-2 for fourc::particles#10 — the rule holds, the Signal does not, and the
# outcome is worse than the one claimed.
#
# Claimed:  "an impactor moving outside the original bounding box triggers
#           'particle out of domain' from 4C particle engine — simulation aborts
#           mid-run."
# Observed: nothing aborts.  Upstream particle_sph_2d_pdbody_gravity.4C.yaml is
#           driven by gravity along +x; pulling the box's +x face in from 20.0 to
#           17.0 leaves every particle inside at setup except the outermost wall
#           column, and then, as the body travels, 4C quietly DELETES the
#           particles that cross the face and keeps integrating.  The only trace
#           is
#             on processor 0 removed 3 particle(s) being outside the
#             computational domain!
#           which names no id, no position, no step and no phase.  It fires twice
#           after the run has started — those are particles that WERE inside the
#           box at t=0 and left it while moving, which is exactly the motion
#           range the entry is about — and all 3000 steps still run.  The string
#           "particle out of domain" does not appear anywhere.
#
# The failure only surfaces at the very end, as eight wrong result values, so a
# too-small box does not stop a run: it silently deletes part of the model and
# hands back an answer.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream particle_sph_2d_pdbody_gravity.4C.yaml) || exit 3
OLDBOX='  DOMAINBOUNDINGBOX: "-20.0 -20.0 -0.01 20.0 20.0 0.01"'
grep -qF "$OLDBOX" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  GRAVITY_ACCELERATION: "9810.0 0.0 0.0"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/bigbox.yaml"
python3 - "$BASE" "$TMP/smallbox.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = '  DOMAINBOUNDINGBOX: "-20.0 -20.0 -0.01 20.0 20.0 0.01"'
new = '  DOMAINBOUNDINGBOX: "-20.0 -20.0 -0.01 17.0 20.0 0.01"'
assert old in t
open(sys.argv[2], "w").write(t.replace(old, new))
PY

probe BIGBOX   "$TMP/bigbox.yaml"
probe SMALLBOX "$TMP/smallbox.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BIGBOX.log"
echo "BIGBOX_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BIGBOX.log")"
echo "BIGBOX_REMOVAL_EVENTS=$(grep -c 'being outside the computational domain' "$TMP/BIGBOX.log")"

grep -m1 -F "on processor 0 removed 120 particle(s) being outside the computational domain!" "$TMP/SMALLBOX.log"
grep -m1 -F "on processor 0 removed 3 particle(s) being outside the computational domain!" "$TMP/SMALLBOX.log"

# Split the removals into "was already outside at setup" and "walked out later".
python3 - "$TMP/SMALLBOX.log" <<'PY'
import sys
lines = open(sys.argv[1], errors="replace").read().split("\n")
first_step = next(i for i, l in enumerate(lines)
                  if "Number of pd_neighbor_pairs in peridynamic evaluation" in l)
rem = [i for i, l in enumerate(lines) if "being outside the computational domain" in l]
print("SMALLBOX_REMOVAL_EVENTS=%d" % len(rem))
print("SMALLBOX_SETUP_REMOVAL_EVENTS=%d" % len([i for i in rem if i < first_step]))
print("SMALLBOX_MIDRUN_REMOVAL_EVENTS=%d" % len([i for i in rem if i > first_step]))
PY

# The run is not stopped by any of it.
echo "SMALLBOX_STEPS_RUN=$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/SMALLBOX.log")"
echo "CLAIMED_PARTICLE_OUT_OF_DOMAIN_TEXT=$(grep -ci 'particle out of domain' "$TMP/SMALLBOX.log")"
echo "SMALLBOX_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SMALLBOX.log")"
grep -m1 -oE "4C_utils_result_test\.cpp, line [0-9]+" "$TMP/SMALLBOX.log"
if [ "$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/SMALLBOX.log")" = \
     "$(grep -c 'Number of pd_neighbor_pairs in peridynamic evaluation' "$TMP/BIGBOX.log")" ]; then
  echo "SIMULATION_ABORTED_MID_RUN=no"; else echo "SIMULATION_ABORTED_MID_RUN=yes"; fi
exit 0
