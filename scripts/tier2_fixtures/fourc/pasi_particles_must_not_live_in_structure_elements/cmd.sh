#!/bin/bash
# Tier-2 for fourc::pasi#4 — the two fields really do need their own geometry
# blocks, and the parser complains about the wrong thing.
#
# Claimed:  "putting particles inside STRUCTURE GEOMETRY raises 'unexpected
#           particle entry in structural mesh' at parse".
# Observed: a particle line dropped into STRUCTURE ELEMENTS is read as an
#           element line, so the parser trips on the very first token while
#           trying to read an element id:
#             Could not parse 'TYPE' as an integer value.
#           from core/io/src/4C_io_value_parser.cpp.  Nothing in that mentions
#           particles, the PARTICLES section, or a structural mesh — it looks
#           like an ordinary malformed-element error, which is the wrong place to
#           go looking.  There is no 'unexpected particle entry' string in 4C.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream pasi_twoway_norelax_particle_dem_1d_normalcontact_linspring_walldiscretcond.4C.yaml) || exit 3
grep -q "^STRUCTURE ELEMENTS:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^PARTICLES:"          "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/separate.yaml"
python3 - "$BASE" "$TMP/mixed.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
assert "STRUCTURE ELEMENTS:\n" in t
open(sys.argv[2], "w").write(
    t.replace("STRUCTURE ELEMENTS:\n",
              'STRUCTURE ELEMENTS:\n  - "TYPE phase1 POS 0.0 0.0 0.5"\n', 1))
PY
echo "PARTICLE_LINE_INJECTED=$(( $(grep -c 'TYPE phase1 POS 0.0 0.0 0.5' "$TMP/mixed.yaml") \
                               - $(grep -c 'TYPE phase1 POS 0.0 0.0 0.5' "$TMP/separate.yaml") ))"

probe SEPARATE "$TMP/separate.yaml"
probe MIXED    "$TMP/mixed.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SEPARATE.log"
echo "SEPARATE_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/SEPARATE.log")"

grep -m1 -F "Could not parse 'TYPE' as an integer value." "$TMP/MIXED.log"
grep -m1 -oE "4C_io_value_parser\.cpp, line [0-9]+" "$TMP/MIXED.log"
echo "CLAIMED_UNEXPECTED_PARTICLE_ENTRY_TEXT=$(grep -ci 'unexpected particle entry' "$TMP/MIXED.log")"
if grep -qiE "Could not parse 'TYPE'.*(particle|PARTICLES|structural mesh)" "$TMP/MIXED.log"; then
  echo "DIAGNOSTIC_MENTIONS_PARTICLES=yes"
else
  echo "DIAGNOSTIC_MENTIONS_PARTICLES=no"
fi
exit 0
