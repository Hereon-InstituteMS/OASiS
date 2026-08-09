#!/bin/bash
# Tier-2 for fourc::brownian_dynamics#4 (and #5, the missing-key SIGFPE).
#
# Claimed: a Brownian run is deterministic WITHIN a build but its stored
#          reference values do not transfer BETWEEN builds; and dropping
#          VISCOSITY or BROWNDYNPROB kills the run with no 4C message.
#
# The portability half is measured with a positive control, because "4C's own
# decks fail" is only meaningful if a comparable deck passes on the same binary.
#
# T2_MUTATE=1 removes the pathology from the two dropped-key decks -- they are
# rebuilt untouched -- so no SIGFPE occurs and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream beam3eb_backweuler_browndyn_singlefil.4C.yaml) || exit 3
grep -q "BROWNDYNPROB: true" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "RANDSEED: 1"        "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_lost_its_seed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
def w(n, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
w("novisc", src.replace("  VISCOSITY: 0.001\n", ""))
w("noprob", src.replace("  BROWNDYNPROB: true\n", ""))
open(tmp + "/rep.yaml", "w").write(src)
PY

# --- deterministic WITHIN the build ---------------------------------------
probe RUN1 "$BASE"
probe RUN2 "$TMP/rep.yaml"
grep "actresult=" "$TMP/RUN1.log" > "$TMP/v1"
grep "actresult=" "$TMP/RUN2.log" > "$TMP/v2"
echo "RESULT_LINES=$(wc -l < "$TMP/v1")"
cmp -s "$TMP/v1" "$TMP/v2" && echo "DETERMINISTIC_WITHIN_BUILD=yes" || echo "DETERMINISTIC_WITHIN_BUILD=no"

# --- stored reference values, across 4C's own browndyn decks --------------
n=0; repro=0
for f in "$DECKS"/beam*browndyn*.4C.yaml; do
  [ -f "$f" ] || continue
  [ "$n" -ge 10 ] && break
  n=$((n+1))
  probe "BD$n" "$f" > /dev/null
  if [ "$(grep -c 'is WRONG' "$TMP/BD$n.log")" = 0 ]; then
    repro=$((repro+1))
  fi
done
echo "BROWNDYN_DECKS_RUN=$n"
echo "BROWNDYN_DECKS_REPRODUCING_THEIR_REFERENCE=$repro"
# The point is not that they all fail -- some do reproduce -- but that the
# outcome is deck-dependent on one and the same binary, so a failing reference
# value is not evidence that YOUR deck is wrong.
echo "BROWNDYN_REFERENCE_OUTCOME_IS_MIXED=$([ "$repro" -gt 0 ] && [ "$repro" -lt "$n" ] && echo yes || echo no)"
# and the failure is at the scale of the answer, not at roundoff
python3 - "$TMP/RUN1.log" <<'PY'
import re, sys
rows = re.findall(r"actresult=\s*(-?[\d.]+e[+-]\d+), givenresult=\s*(-?[\d.]+e[+-]\d+)", open(sys.argv[1]).read())
if not rows:
    print("MISMATCH_IS_LEADING_DIGIT=unknown")
else:
    big = sum(1 for a, g in rows
              if abs(float(a) - float(g)) > 0.1 * max(abs(float(a)), abs(float(g)), 1e-30))
    print("MISMATCH_IS_LEADING_DIGIT=%s" % ("yes" if big == len(rows) else "no"))
PY
# POSITIVE CONTROL: a non-Brownian beam deck on the same binary.
CTRL=$(upstream beam3r_line2_elastoplastic_axialNeum_woIsohard.4C.yaml) || exit 3
probe CONTROL "$CTRL"
echo "NON_BROWNIAN_BEAM_DECK_REPRODUCES=$([ "$(grep -c 'is WRONG' "$TMP/CONTROL.log")" = 0 ] && echo yes || echo no)"

# --- the two keys with no usable default ----------------------------------
probe NOVISC "$TMP/novisc.yaml"
probe NOPROB "$TMP/noprob.yaml"
for L in NOVISC NOPROB; do
  echo "${L}_IS_A_SIGNAL=$(grep -c 'Floating point exception' "$TMP/$L.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
  echo "${L}_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/$L.log")"
  echo "${L}_NAMES_THE_KEY=$(grep -cE 'VISCOSITY|BROWNDYNPROB|damping coefficient' "$TMP/$L.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
done
grep -m1 -F "Signal code: Floating point divide-by-zero (3)" "$TMP/NOVISC.log"
exit 0
