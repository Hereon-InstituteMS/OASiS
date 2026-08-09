#!/bin/bash
# Tier-2 for fourc::ehl#5 — the squeeze-film term is real, but the key the entry
# tells you to set does not exist.
#
# Claimed:  "Set TRANSIENT: true in LUBRICATION DYNAMIC for time-accurate
#            problems."
# Observed: there is no TRANSIENT key in LUBRICATION DYNAMIC.  Writing it aborts
#           at parse time in core/io/src/4C_io_input_spec_builders.cpp line 633
#           with "Could not match this input", and 4C helpfully echoes the whole
#           offending section back.  The real key is ADD_SQUEEZE_TERM (bool,
#           default false), which upstream ehl3d_mixed.4C.yaml already sets to
#           true; turning it off changes the physics enough that the run dies of
#           SIGFPE inside the contact regularisation rather than finishing.
#
# So: an author following the entry gets a hard parse error before anything is
# built.  The diagnostic is good — it names TRANSIENT under "The following data
# remains unused:" and enumerates every parameter the section really accepts,
# ADD_SQUEEZE_TERM included — but the entry sent them to a key that never was.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ehl3d_mixed.4C.yaml) || exit 3
grep -q '  ADD_SQUEEZE_TERM: true' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_squeeze_key_changed"; exit 3; }
grep -q 'TRANSIENT' "$BASE" \
  && { echo "FIXTURE_ABORT=upstream_now_has_a_transient_key"; exit 3; }

# The claimed key, and the real one.
CLAIMED_KEY='  TRANSIENT: true'
SQUEEZE_SETTING=false

cp "$BASE" "$TMP/squeeze_on.yaml"
sed "s/  ADD_SQUEEZE_TERM: true/  ADD_SQUEEZE_TERM: true\n$CLAIMED_KEY/" "$BASE" > "$TMP/claimedkey.yaml"
sed "s/  ADD_SQUEEZE_TERM: true/  ADD_SQUEEZE_TERM: $SQUEEZE_SETTING/"   "$BASE" > "$TMP/squeeze_off.yaml"
grep -m1 '  ADD_SQUEEZE_TERM:' "$TMP/squeeze_off.yaml" | tr -d ' ' | sed 's/^/OFF_ARM_/'
echo "CLAIMED_KEY_IN_DECK=$(grep -c 'TRANSIENT' "$TMP/claimedkey.yaml")"

probe SQUEEZE_ON  "$TMP/squeeze_on.yaml"
probe CLAIMEDKEY  "$TMP/claimedkey.yaml"
probe SQUEEZE_OFF "$TMP/squeeze_off.yaml"

grep -m1 -F "OK (7)" "$TMP/SQUEEZE_ON.log"
grep -m1 -F "processor 0 finished normally" "$TMP/SQUEEZE_ON.log"
grep -m1 -F "Could not match this input" "$TMP/CLAIMEDKEY.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/CLAIMEDKEY.log"
grep -m1 -F "LUBRICATION DYNAMIC:" "$TMP/CLAIMEDKEY.log"
grep -m1 -F "The following data remains unused:" "$TMP/CLAIMEDKEY.log"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/SQUEEZE_OFF.log"

# The parse failure happens before anything is built.
echo "CLAIMEDKEY_FIELDS_BUILT=$(grep -c 'fill_complete() on discretization' "$TMP/CLAIMEDKEY.log")"
# The diagnostic is good: it names the offending key and enumerates every
# parameter the section really accepts, ADD_SQUEEZE_TERM among them.
echo "CLAIMEDKEY_NAMES_THE_UNUSED_KEY=$(grep -c 'The following data remains unused:' "$TMP/CLAIMEDKEY.log")"
echo "CLAIMEDKEY_ENUMERATES_REAL_KEY=$(grep -c "Matched parameter 'ADD_SQUEEZE_TERM'" "$TMP/CLAIMEDKEY.log")"
# And the real key is load-bearing: turning it off breaks the run.
echo "SQUEEZE_OFF_DIES=$(grep -c 'Signal: Floating point exception' "$TMP/SQUEEZE_OFF.log")"
echo "SQUEEZE_OFF_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/SQUEEZE_OFF.log")"
exit 0
