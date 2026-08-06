#!/bin/bash
# Tier-2 for fourc::particle_dem#1 — MAX_RADIUS defaults to 0.0 and 0.0 is
# rejected, so it is mandatory in every DEM deck; MIN_RADIUS above it has its
# own message.
#
# T2_MUTATE=1 removes both edits, so both decks are the untouched upstream one,
# nothing aborts and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
grep -q "MAX_RADIUS: 0.01" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
def w(n, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
w("nomax", src.replace("  MAX_RADIUS: 0.01\n", ""))                       # falls back to 0.0
w("minmax", src.replace("  MAX_RADIUS: 0.01\n",
                        "  MIN_RADIUS: 0.02\n  MAX_RADIUS: 0.01\n"))
PY

probe NOMAX  "$TMP/nomax.yaml"
probe MINMAX "$TMP/minmax.yaml"
probe BASE   "$BASE"

grep -m1 -F "non-positive maximum allowed particle radius!" "$TMP/NOMAX.log"   && echo "OMITTED_MAX_ABORTS=yes" || echo "OMITTED_MAX_ABORTS=no"
grep -m1 -F "minimum allowed particle radius larger than maximum allowed particle radius!" "$TMP/MINMAX.log"   && echo "MIN_ABOVE_MAX_ABORTS=yes" || echo "MIN_ABOVE_MAX_ABORTS=no"
grep -m1 -F "4C_particle_interaction_dem.cpp" "$TMP/NOMAX.log" | sed 's/.*\(4C_particle_interaction_dem.cpp\).*/SOURCE_FILE=\1/'
# Both abort BEFORE the first time step: no TIME: line is ever printed.
echo "NOMAX_STEPS_PRINTED=$(grep -c '^TIME:' "$TMP/NOMAX.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log" && echo "BASE_CLEAN=yes"
# Every upstream DEM deck sets MAX_RADIUS -- it is not optional in practice.
echo "UPSTREAM_DECKS_WITHOUT_MAX_RADIUS=$(for f in "$DECKS"/particle_dem_*.4C.yaml; do grep -qE '^  MAX_RADIUS:' "$f" || echo x; done | wc -l)"
exit 0
