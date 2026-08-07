#!/bin/bash
# Tier-2 for fourc::xfem_fluid#2 -- an XFEM fluid with no interface description
# does not "revert to standard FEM".  It dies inside the cut wizard.
#
# Claimed: result matches a non-enriched reference exactly and the XFEM
#          diagnostic prints `0 enriched elements`.
# Observed: no such diagnostic exists.  Delete the only XFEM coupling condition
#          from the upstream level-set deck and 4C aborts in
#          Cut::CutWizard::safety_checks with "You have to call PrepareCut()
#          before you can call the Cut-routine" -- an internal-API message that
#          never mentions the missing condition.  There is no silent fallback.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream xfluid_ls_neumann_inflow_stab.4C.yaml) || exit 3
grep -q "^DESIGN XFEM LEVELSET NEUMANN VOL CONDITIONS:" "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/withcond.yaml"
python3 - "$BASE" "$TMP/nocond.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
block = """DESIGN XFEM LEVELSET NEUMANN VOL CONDITIONS:
  - E: 1
    COUPLINGID: 1
    LEVELSETFIELDNO: 2
    BOOLEANTYPE: "none"
    COMPLEMENTARY: false
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [-5, -5, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
    INFLOW_STAB: true
"""
assert block in t, "upstream deck no longer carries the level-set Neumann coupling"
open(sys.argv[2], "w").write(t.replace(block, ""))
PY

probe WITHCOND "$TMP/withcond.yaml"
probe NOCOND   "$TMP/nocond.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/WITHCOND.log"
grep -m1 -F "You have to call PrepareCut() before you can call the Cut-routine" "$TMP/NOCOND.log"
grep -m1 -F "4C_cut_cutwizard.cpp" "$TMP/NOCOND.log"
# It never reaches a result test, so it cannot "match a standard reference".
echo "NOCOND_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOCOND.log")"
echo "CLAIMED_ENRICHED_ELEMENTS_TEXT=$(grep -ci 'enriched elements' "$TMP/NOCOND.log")"
exit 0
