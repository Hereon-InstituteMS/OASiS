#!/bin/bash
# Tier-2 for fourc::particle_dem#2 — the DEM normal stiffness is an
# exclusive-or, and all three ways of getting it wrong print the SAME sentence.
#
# Claimed: set EITHER NORMAL_STIFF, OR REL_PENETRATION together with
#          MAX_VELOCITY. Both, neither, or REL_PENETRATION without MAX_VELOCITY
#          abort with one identical message that does not say which half is
#          wrong.
#
# The two upstream decks are the two legal halves: ..._stiffset sets
# NORMAL_STIFF, ..._stiffauto sets REL_PENETRATION + MAX_VELOCITY. Both pass
# untouched. Each of the three illegal combinations is built from stiffauto by
# a single edit.
#
# T2_MUTATE=1 removes the pathology: the three broken decks are built without
# their edit, i.e. they are three copies of the working stiffauto deck. The
# abort then never happens, XOR_ABORTS drops to 0 and every EXIT_* becomes 0,
# so the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

AUTO=$(upstream particle_dem_1d_normalcontact_linspring_stiffauto.4C.yaml) || exit 3
SET=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml)   || exit 3
grep -q "REL_PENETRATION: 0.05" "$AUTO" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "NORMAL_STIFF: 3.5e-05" "$SET"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$AUTO" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
def w(name, text):
    open("%s/%s.yaml" % (tmp, name), "w").write(src if mutate else text)
# both halves at once
w("both",   src.replace("  REL_PENETRATION: 0.05\n",
                        "  REL_PENETRATION: 0.05\n  NORMAL_STIFF: 3.5e-05\n"))
# neither half
w("none",   src.replace("  REL_PENETRATION: 0.05\n", ""))
# the automatic half without the velocity it needs
w("novmax", src.replace("  MAX_VELOCITY: 0.03\n", ""))
PY

probe STIFFAUTO "$AUTO"
probe STIFFSET  "$SET"
probe BOTH      "$TMP/both.yaml"
probe NONE      "$TMP/none.yaml"
probe NOVMAX    "$TMP/novmax.yaml"

MSG="specify either the relative penetration along with the maximum velocity, or the normal stiffness, but neither both nor none of them!"
n=0
for L in BOTH NONE NOVMAX; do
  if grep -qF "$MSG" "$TMP/$L.log"; then n=$((n+1)); echo "ABORTED_$L=yes"; else echo "ABORTED_$L=no"; fi
done
echo "XOR_ABORTS=$n"
grep -m1 -F "$MSG" "$TMP/BOTH.log"
grep -m1 -F "4C_particle_interaction_dem_contact_normal.cpp" "$TMP/BOTH.log" | sed 's/.*\(4C_particle_interaction_dem_contact_normal.cpp\).*/SOURCE_FILE=\1/'

# The message is byte-identical in all three cases, so it cannot tell you which
# half you got wrong. Count the distinct sentences.
echo "DISTINCT_MESSAGES=$(for L in BOTH NONE NOVMAX; do grep -m1 -F "$MSG" "$TMP/$L.log"; done | sort -u | wc -l)"

# Both legal halves run clean.
echo "LEGAL_HALVES_OK=$(grep -lc 'OK (' "$TMP/STIFFAUTO.log" "$TMP/STIFFSET.log" 2>/dev/null | wc -l)"
grep -m1 -F "processor 0 finished normally" "$TMP/STIFFAUTO.log" && echo "STIFFAUTO_CLEAN=yes"
grep -m1 -F "processor 0 finished normally" "$TMP/STIFFSET.log"  && echo "STIFFSET_CLEAN=yes"
exit 0
