#!/bin/bash
# Tier-2 for fourc::particle_dem#3 — DEM computes a critical time step, checks
# it every step, and only WARNS.
#
# Claimed: exceeding dt_crit prints 'Warning: time step <dt> larger than
#          critical time step <dtcrit>!' once per step and never aborts; the run
#          completes and only the result verdicts show the damage.
#
# T2_MUTATE=1 removes the pathology: TIMESTEP is left at the deck's own value,
# the warning is never printed, WARN_LINES drops to 0 and VERDICTS_WRONG to 0,
# so the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
grep -q "TIMESTEP: 0.001" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
# 20x the deck's step, and only 50 of them so the log stays readable
big = src.replace("  TIMESTEP: 0.001", "  TIMESTEP: 0.02").replace("  NUMSTEP: 1000", "  NUMSTEP: 50")
ok  = src.replace("  NUMSTEP: 1000", "  NUMSTEP: 50")
open(tmp + "/bigdt.yaml", "w").write(ok if mutate else big)
PY

probe BIGDT "$TMP/bigdt.yaml"

# The warning exists, quotes both numbers, and fires once per step (50 steps).
grep -m1 -E "^Warning: time step .* larger than critical time step .*!$" "$TMP/BIGDT.log"
echo "WARN_LINES=$(grep -c 'larger than critical time step' "$TMP/BIGDT.log")"
echo "WARN_IS_ONE_PER_STEP=$([ "$(grep -c 'larger than critical time step' "$TMP/BIGDT.log")" = 50 ] && echo yes || echo no)"
# It is a warning, not an abort: the time loop reaches the end and the run
# gets as far as the result-test manager.
grep -m1 -F "Checking results of" "$TMP/BIGDT.log" && echo "REACHED_RESULT_TESTS=yes"
echo "ABORTED_IN_THE_TIME_LOOP=$(grep -c 'traveled more than one bin' "$TMP/BIGDT.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# Nothing in the DEM stack raised the alarm any other way.
echo "SOURCE_FILE_MENTIONED=$(grep -c '4C_particle_interaction_dem_contact.cpp' "$TMP/BIGDT.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# The only real symptom is the verdicts.
echo "VERDICTS_WRONG=$(grep -c 'is WRONG' "$TMP/BIGDT.log")"
exit 0
