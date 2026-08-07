#!/bin/bash
# Tier-2 for fourc::pasi#6 (and #7, the time-step ownership) — COUPLING defaults
# to ONE-WAY, and a TIMESTEP written in PARTICLE DYNAMIC is inert.
#
# T2_MUTATE=1 removes both edits; COUPLING stays, no conflicting particle step
# is added, the iteration loop is present and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream pasi_twoway_norelax_particle_dem_1d_normalcontact_linspring_walldiscretcond.4C.yaml) || exit 3
grep -q "COUPLING: partitioned_twowaycoup" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# Every upstream PASI deck leaves PARTICLE DYNAMIC without a TIMESTEP.
echo "PASI_DECKS_WITH_A_PARTICLE_TIMESTEP=$(for f in "$DECKS"/pasi_*.4C.yaml; do python3 -c "
import sys,re
t=open(sys.argv[1]).read()
seg=re.split(r'^PARTICLE DYNAMIC:$', t, flags=re.M)
print('x' if len(seg)>1 and re.search(r'^  TIMESTEP:', re.split(r'^[A-Z][A-Z ]+', seg[1], flags=re.M)[0], flags=re.M) else '', end='')
" "$f"; done | wc -c)"

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
def w(n, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
w("nocoup",  src.replace("  COUPLING: partitioned_twowaycoup\n", ""))
# a particle-field step ten times the PASI one
w("dtclash", src.replace("PARTICLE DYNAMIC:\n", "PARTICLE DYNAMIC:\n  TIMESTEP: 0.005\n"))
PY

probe BASE    "$BASE"
probe NOCOUP  "$TMP/nocoup.yaml"
probe DTCLASH "$TMP/dtclash.yaml"

grep -m1 -E "^OK \(" "$TMP/BASE.log" && echo "BASE_PASSES=yes"

# --- COUPLING defaults to one-way, silently -------------------------------
echo "NOCOUP_MENTIONS_COUPLING=$(grep -ci 'coupl' "$TMP/NOCOUP.log")"
echo "NOCOUP_VERDICTS_WRONG=$(grep -c 'is WRONG' "$TMP/NOCOUP.log")"
echo "NOCOUP_DEGRADES_SILENTLY=$([ "$(grep -c 'is WRONG' "$TMP/NOCOUP.log")" -gt 0 ] && [ "$(grep -ci 'coupl' "$TMP/NOCOUP.log")" = 0 ] && echo yes || echo no)"
# The one observable that separates one-way from two-way: a two-way run has a
# fixed-point loop to report, a one-way run has none. Counted on both.
echo "TWOWAY_ITERATION_LINES=$(grep -ci 'iteration' "$TMP/BASE.log")"
echo "ONEWAY_ITERATION_LINES=$(grep -ci 'iteration' "$TMP/NOCOUP.log")"
echo "ITERATION_COUNT_SEPARATES_THEM=$([ "$(grep -ci 'iteration' "$TMP/BASE.log")" -gt 0 ] && [ "$(grep -ci 'iteration' "$TMP/NOCOUP.log")" = 0 ] && echo yes || echo no)"

# --- the time step belongs to PASI DYNAMIC --------------------------------
grep -m1 -F "Overview of chosen time stepping:" "$TMP/DTCLASH.log" && echo "TIMESTEP_TABLE_IS_PRINTED=yes"
grep -m1 -F "currently equal for both structure and particle field" "$TMP/DTCLASH.log"
# The Particles column shows the PASI step, not the one written in PARTICLE DYNAMIC.
python3 - "$TMP/DTCLASH.log" <<'PY'
import re, sys
m = re.search(r"^\s*Timestep:\s+(\S+)\s+(\S+)\s+(\S+)\s*$", open(sys.argv[1]).read(), re.M)
if not m:
    print("TIMESTEP_ROW_PARSED=no")
else:
    pasi, part, struct = (float(x) for x in m.groups())
    print("TIMESTEP_ROW_PARSED=yes")
    print("PARTICLE_COLUMN_EQUALS_PASI=%s" % ("yes" if part == pasi else "no"))
    print("PARTICLE_COLUMN_EQUALS_THE_KEY_WE_WROTE=%s" % ("yes" if part == 0.005 else "no"))
PY
grep -E "is CORRECT|is WRONG" "$TMP/BASE.log"    > "$TMP/v_base"
grep -E "is CORRECT|is WRONG" "$TMP/DTCLASH.log" > "$TMP/v_clash"
echo "VERDICT_LINES=$(wc -l < "$TMP/v_base")"
cmp -s "$TMP/v_base" "$TMP/v_clash" && echo "PARTICLE_TIMESTEP_IS_INERT=yes" || echo "PARTICLE_TIMESTEP_IS_INERT=no"
echo "CLASH_DECK_REALLY_CARRIES_THE_KEY=$(grep -c 'TIMESTEP: 0.005' "$TMP/dtclash.yaml" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
exit 0
