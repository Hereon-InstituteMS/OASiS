#!/bin/bash
# Tier-2 for fourc::particle_dem#9 — the adhesion keys are negative sentinels,
# and their messages describe the SENTINEL rather than the omission; one key
# wants a negative value and reports a positive one as the error.
#
# T2_MUTATE=1 removes every edit; the adhesion block is complete, nothing
# aborts and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

AD=$(upstream particle_dem_1d_adhesion_VdWDMT.4C.yaml) || exit 3
grep -q "ADHESION_MAX_CONTACT_PRESSURE: -300" "$AD" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$AD" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
def w(n, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
w("nodist",     src.replace("  ADHESION_DISTANCE: 0.0025\n", ""))
w("nohamaker",  src.replace("  ADHESION_HAMAKER: 4e-16\n", ""))
w("pospress",   src.replace("  ADHESION_MAX_CONTACT_PRESSURE: -300",
                            "  ADHESION_MAX_CONTACT_PRESSURE: 300"))
# a legal but easy-to-miss value: zero adhesion distance is ACCEPTED
w("zerodist",   src.replace("  ADHESION_DISTANCE: 0.0025", "  ADHESION_DISTANCE: 0.0"))
PY

probe NODIST    "$TMP/nodist.yaml"
probe NOHAMAKER "$TMP/nohamaker.yaml"
probe POSPRESS  "$TMP/pospress.yaml"
probe ZERODIST  "$TMP/zerodist.yaml"
probe ADBASE    "$AD"

# Omitting a key reports the SENTINEL, i.e. it says "negative" about a key you
# never wrote.
grep -m1 -F "negative adhesion distance!" "$TMP/NODIST.log"   && echo "OMITTED_DISTANCE_READS_AS_NEGATIVE=yes" || echo "OMITTED_DISTANCE_READS_AS_NEGATIVE=no"
grep -m1 -F "negative hamaker constant!" "$TMP/NOHAMAKER.log"   && echo "OMITTED_HAMAKER_READS_AS_NEGATIVE=yes" || echo "OMITTED_HAMAKER_READS_AS_NEGATIVE=no"
# Neither message names the key as missing.
echo "MESSAGES_SAY_MISSING=$(cat "$TMP/NODIST.log" "$TMP/NOHAMAKER.log" | grep -ciE 'not set|missing|required' | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# The sign trap: a POSITIVE maximum contact pressure is the error.
grep -m1 -F "positive adhesion maximum contact pressure!" "$TMP/POSPRESS.log"   && echo "POSITIVE_PRESSURE_IS_THE_ERROR=yes" || echo "POSITIVE_PRESSURE_IS_THE_ERROR=no"
# Zero adhesion distance is accepted, is not a no-op, and warns about nothing.
echo "ZERODIST_ABORTS=$(grep -c 'PROC 0 ERROR in /.*particle' "$TMP/ZERODIST.log")"
echo "ZERODIST_VERDICTS_WRONG=$(grep -c 'is WRONG' "$TMP/ZERODIST.log")"
echo "ZERODIST_IS_SILENT_BUT_NOT_INERT=$([ "$(grep -c 'is WRONG' "$TMP/ZERODIST.log")" -gt 0 ] && echo yes || echo no)"
grep -m1 -E "^OK \(" "$TMP/ADBASE.log" && echo "ADBASE_PASSES=yes"
exit 0
