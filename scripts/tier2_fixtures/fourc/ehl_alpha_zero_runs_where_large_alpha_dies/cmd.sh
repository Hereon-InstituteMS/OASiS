#!/bin/bash
# Tier-2 for fourc::ehl#3 — "start with constant viscosity (alpha = 0) to verify
# the setup, then ramp alpha" is good advice, and this measures the window it
# buys you on upstream ehl_bearing_barus.4C.yaml (PreVisCoeff = alpha = 0.0112).
#
#   alpha = 0        : the Barus law degenerates to constant viscosity.  Complete
#                      run, both time steps, all four result tests evaluated —
#                      they report WRONG only because the deck's reference values
#                      belong to the piezoviscous law, which is exactly what an
#                      alpha = 0 sanity run is for.
#   alpha = 3 x      : still completes.
#   alpha = 10 x     : dead.  SIGFPE (shell status 136) before the first time
#                      step finishes; no 4C error line, no result-test section.
#
# So the failure is not "Newton diverges in step 1" in any reportable sense — the
# process is killed inside the element evaluation and 4C never gets to say
# anything.  The one usable signal is that the run produced no
# "Checking results of" line at all.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ehl_bearing_barus.4C.yaml) || exit 3
grep -q '      PreVisCoeff: 0.0112' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_previscoeff_changed"; exit 3; }

# The cold-start ladder.  COLD_START_ALPHA is the value an author would use to
# check the setup; DEAD_ALPHA is the one that kills it.
COLD_START_ALPHA=0.0
RAMP_ALPHA=0.0336
DEAD_ALPHA=0.112

sed "s/      PreVisCoeff: 0.0112/      PreVisCoeff: $COLD_START_ALPHA/" "$BASE" > "$TMP/alpha0.yaml"
sed "s/      PreVisCoeff: 0.0112/      PreVisCoeff: $RAMP_ALPHA/"       "$BASE" > "$TMP/ramp.yaml"
sed "s/      PreVisCoeff: 0.0112/      PreVisCoeff: $DEAD_ALPHA/"       "$BASE" > "$TMP/dead.yaml"
grep -m1 '      PreVisCoeff:' "$TMP/alpha0.yaml" | tr -d ' ' | sed 's/^/COLD_ARM_/'
grep -m1 '      PreVisCoeff:' "$TMP/dead.yaml"   | tr -d ' ' | sed 's/^/DEAD_ARM_/'

probe ALPHA0 "$TMP/alpha0.yaml"
probe RAMP   "$TMP/ramp.yaml"
probe DEAD   "$TMP/dead.yaml"

# The alpha = 0 sanity run really did run the physics to the end.
grep -m1 -F "Checking results of 4 tests:" "$TMP/ALPHA0.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/ALPHA0.log"
grep -m1 -F "Result check failed with 4 errors out of 4 tests" "$TMP/ALPHA0.log"
grep -m1 -F "Checking results of 4 tests:" "$TMP/RAMP.log"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/DEAD.log"

echo "ALPHA0_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/ALPHA0.log")"
echo "RAMP_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/RAMP.log")"
echo "ALPHA0_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/ALPHA0.log")"
echo "RAMP_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/RAMP.log")"
echo "DEAD_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/DEAD.log")"
echo "DEAD_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/DEAD.log")"
echo "DEAD_NEWTON_TABLE_ROWS=$(grep -cE '^ +[0-9]+ +[0-9.]+e[-+][0-9]{2}' "$TMP/DEAD.log")"
exit 0
