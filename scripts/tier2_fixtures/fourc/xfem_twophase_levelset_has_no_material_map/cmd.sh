#!/bin/bash
# Tier-2 for fourc::xfem_fluid#3 -- there is no two-phase material map in 4C.
#
# Claimed: 4C aborts with `XFEM: material map needs MAT_NEGATIVE and
#          MAT_POSITIVE`.
# Observed: the condition that would carry such a map, DESIGN XFEM LEVELSET
#          TWOPHASE VOL CONDITIONS, accepts exactly five keys -- E, COUPLINGID,
#          LEVELSETFIELDNO, BOOLEANTYPE, COMPLEMENTARY -- and no material keys
#          at all.  Adding MAT_NEGATIVE/MAT_POSITIVE is rejected as unmatched
#          input, and a *correctly* spelled two-phase condition does not run
#          either: it dies in Element::location_vector with "wrong number of
#          nodes".  Two-phase level-set XFEM is not usable in this build.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfluid_ls_neumann_inflow_stab.4C.yaml) || exit 3
grep -q "^DESIGN XFEM LEVELSET NEUMANN VOL CONDITIONS:" "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

python3 - "$BASE" "$TMP" <<'PY'
import sys, os
t = open(sys.argv[1]).read(); TMP = sys.argv[2]
tail = """    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [-5, -5, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
    INFLOW_STAB: true
"""
assert tail in t, "upstream deck no longer carries the level-set Neumann payload"
tp = t.replace("DESIGN XFEM LEVELSET NEUMANN VOL CONDITIONS:",
               "DESIGN XFEM LEVELSET TWOPHASE VOL CONDITIONS:").replace(tail, "")
open(os.path.join(TMP, "twophase.yaml"), "w").write(tp)
open(os.path.join(TMP, "matmap.yaml"), "w").write(
    tp.replace("    COMPLEMENTARY: false",
               "    COMPLEMENTARY: false\n    MAT_NEGATIVE: 1\n    MAT_POSITIVE: 2"))
PY

probe TWOPHASE "$TMP/twophase.yaml"
probe MATMAP   "$TMP/matmap.yaml"

# The claimed keys are not part of the condition spec at all.
grep -m1 -F "Failed to match condition specification in section 'DESIGN XFEM LEVELSET TWOPHASE VOL CONDITIONS'" "$TMP/MATMAP.log"
grep -m1 -F "MAT_NEGATIVE: 1" "$TMP/MATMAP.log"
# And the correctly-spelled two-phase condition still cannot run.
grep -m1 -F "wrong number of nodes" "$TMP/TWOPHASE.log"
grep -m1 -F "4C_fem_general_element.cpp" "$TMP/TWOPHASE.log"
echo "CLAIMED_MATERIAL_MAP_TEXT=$(grep -ci 'material map needs' "$TMP/TWOPHASE.log" "$TMP/MATMAP.log" | awk -F: '{s+=$2} END {print s+0}')"
exit 0
