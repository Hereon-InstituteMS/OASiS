#!/bin/bash
# Tier-2 for fourc::ssti#0 — SSTI clones twice (structure -> scatra, then
# scatra -> thermo), so the CLONING MATERIAL MAP needs an entry for BOTH hops,
# per material group.  Delete one scatra -> thermo pairing and 4C stops.
#
# Claimed:  'cannot clone material for <field>'.
# Observed: no such string.  What comes out is
#
#     no matching material ID (1) in map
#     core/fem/src/general/utils/4C_fem_general_utils_createdis.hpp
#
# — the SOURCE material ID, not a field name.  That matters when reading the
# log: nothing in the message says "thermo", so an agent has to know that the
# ID it names is the scatra material whose thermo counterpart it just removed.
#
# Also note the entry's arithmetic: it says the map "must contain exactly two
# SRC/TAR entries".  The upstream deck has four, because there are two material
# groups and each needs both hops.  Asserted.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssti_mono_3D_3hex8_elch_s2i_butlervolmerthermo_growthlaw.4C.yaml) || exit 3
grep -q "^CLONING MATERIAL MAP:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/both.yaml"

python3 - "$BASE" "$TMP/thermo_gone.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
hop = ('  - SRC_FIELD: "scatra"\n'
       '    SRC_MAT: 1\n'
       '    TAR_FIELD: "thermo"\n'
       '    TAR_MAT: 6\n')
if hop not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(t.replace(hop, "", 1))
PY
[ -f "$TMP/thermo_gone.yaml" ] || exit 3
# The structure -> scatra hop for the same group is still there.
grep -q 'TAR_FIELD: "scatra"' "$TMP/thermo_gone.yaml" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

probe BOTH       "$TMP/both.yaml"
probe THERMOGONE "$TMP/thermo_gone.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BOTH.log"
grep -m1 -F "no matching material ID (1) in map" "$TMP/THERMOGONE.log"
grep -m1 -oF "4C_fem_general_utils_createdis.hpp" "$TMP/THERMOGONE.log"
# The diagnostic never names the field that is missing.
echo "DIAGNOSTIC_NAMES_THERMO=$(grep -c 'no matching material ID (1) in map.*thermo' "$TMP/THERMOGONE.log")"
echo "CLAIMED_CANNOT_CLONE_TEXT=$(grep -ci 'cannot clone material' "$TMP/THERMOGONE.log")"
# Two hops per material group, not "exactly two entries" overall.
echo "UPSTREAM_CLONING_ENTRIES=$(grep -c 'SRC_FIELD:' "$BASE")"
exit 0
