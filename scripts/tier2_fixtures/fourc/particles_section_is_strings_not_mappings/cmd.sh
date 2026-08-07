#!/bin/bash
# Tier-2 for fourc::particles#12 — the PARTICLES section is a list of STRINGS,
# the phase-name set is closed, and global ids follow file order from 0.
#
# T2_MUTATE=1 removes every edit; the PARTICLES section is the upstream one,
# nothing is refused and the fixture must go red.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

MUTATE="${T2_MUTATE:-0}"

BASE=$(upstream particle_dem_1d_normalcontact_linspring_stiffset.4C.yaml) || exit 3
grep -q '"TYPE phase1 POS -0.015 0.0 0.0"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" "$MUTATE" <<'PY'
import sys
src, tmp, mutate = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3] == "1"
def w(n, t): open("%s/%s.yaml" % (tmp, n), "w").write(src if mutate else t)
# the mapping shape an agent naturally writes
w("mapping", src.replace('  - "TYPE phase1 POS -0.015 0.0 0.0"',
    "  - TYPE: phase1\n    POS: [-0.015, 0.0, 0.0]\n    MAT: 1"))
# an invented phase name
w("badphase", src.replace("TYPE phase1 POS -0.015", "TYPE fluidphase POS -0.015")
                 .replace('PHASE_TO_DYNLOADBALFAC: "phase1 1.0"',
                          'PHASE_TO_DYNLOADBALFAC: "phase1 1.0 fluidphase 1.0"')
                 .replace('PHASE_TO_MATERIAL_ID: "phase1 1"',
                          'PHASE_TO_MATERIAL_ID: "phase1 1 fluidphase 1"'))
# ids follow FILE ORDER: swap the two lines and the result tests swap with them
a, b = '  - "TYPE phase1 POS -0.015 0.0 0.0"', '  - "TYPE phase1 POS 0.015 0.0 0.0"'
assert a in src and b in src
w("swapped", src.replace(a, "@A@").replace(b, a).replace("@A@", b))
PY

probe MAPPING  "$TMP/mapping.yaml"
probe BADPHASE "$TMP/badphase.yaml"
probe SWAPPED  "$TMP/swapped.yaml"
probe BASE     "$BASE"

grep -m1 -E "^OK \(" "$TMP/BASE.log" && echo "BASE_PASSES=yes"
# A mapping-shaped entry is refused ...
echo "MAPPING_REFUSED=$([ "$(grep -c 'PROC 0 ERROR' "$TMP/MAPPING.log")" -gt 0 ] && echo yes || echo no)"
# ... but the message is about the SECTION, not about particle syntax.
echo "MAPPING_MESSAGE_EXPLAINS_THE_GRAMMAR=$(grep -ciE 'TYPE .*POS|expected TYPE|particle syntax' "$TMP/MAPPING.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
grep -m1 -F "Yaml node does not contain a string. This legacy function is only meant for strings." "$TMP/MAPPING.log"
# The message names neither the section nor particles; only the stack frame does.
echo "MAPPING_MESSAGE_NAMES_THE_SECTION=$(sed -n '/PROC 0 ERROR/,/^---/p' "$TMP/MAPPING.log" | grep -c 'PARTICLES' | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "MAPPING_ONLY_THE_STACK_NAMES_PARTICLES=$(grep -c 'Particle::read_particles' "$TMP/MAPPING.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# An invented phase name, declared consistently in BOTH phase maps, is not
# refused at all -- it segfaults with no 4C diagnostic. The drafted claim said
# "you cannot invent one" and implied a clean rejection; this is what happens.
echo "INVENTED_PHASE_SEGFAULTS=$(grep -c 'Segmentation fault' "$TMP/BADPHASE.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
echo "INVENTED_PHASE_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/BADPHASE.log")"
echo "INVENTED_PHASE_NAMED_IN_LOG=$(grep -c 'fluidphase' "$TMP/BADPHASE.log" | sed 's/^0$/no/;s/^[1-9].*/yes/')"
# Ids follow FILE ORDER: swapping two lines swaps which id each test hits, so
# the verdicts break even though the particle set is unchanged.
echo "SWAPPED_VERDICTS_WRONG=$(grep -c 'is WRONG' "$TMP/SWAPPED.log")"
echo "IDS_FOLLOW_FILE_ORDER=$([ "$(grep -c 'is WRONG' "$TMP/SWAPPED.log")" -gt 0 ] && echo yes || echo no)"
echo "SWAPPED_PARTICLE_COUNT_UNCHANGED=$([ "$(grep -c 'TYPE phase1' "$TMP/swapped.yaml")" = "$(grep -c 'TYPE phase1' "$BASE")" ] && echo yes || echo no)"
exit 0
