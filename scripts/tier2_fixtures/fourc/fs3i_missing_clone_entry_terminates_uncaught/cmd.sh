#!/bin/bash
# Tier-2 for fourc::fs3i#1 — FS3I really does need more than one CLONING
# MATERIAL MAP entry, but there is no "cannot clone material for <field>"
# message.  A missing entry kills the process with an uncaught C++ exception and
# 4C's error path never runs.
#
# Upstream fs3i_part_1wc_infperm.4C.yaml declares THREE entries, not the two the
# entry describes: fluid -> scatra1, structure -> scatra2, and fluid -> ale.
# Delete either of the two the entry mentions and the result is identical:
#
#   terminate called after throwing an instance of 'std::out_of_range'
#
# and SIGABRT (shell status 134), with zero "PROC 0 ERROR in" lines, no field
# name, no material id, and no mention of cloning anywhere in the log.  A caller
# grepping for a 4C diagnostic finds nothing at all to key on.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fs3i_part_1wc_infperm.4C.yaml) || exit 3
grep -q '^CLONING MATERIAL MAP:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_cloning_map_changed"; exit 3; }
cp -r "$(dirname "$BASE")/xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_teko_xml"; exit 3; }
cd "$TMP" || exit 3

# The two pathologies.
DROP_SCATRA2_CLONE=yes
DROP_ALE_CLONE=yes

cp "$BASE" "$TMP/full.yaml"
python3 - "$BASE" "$TMP/noscatra2.yaml" "$DROP_SCATRA2_CLONE" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = ('  - SRC_FIELD: "structure"\n    SRC_MAT: 2\n'
       '    TAR_FIELD: "scatra2"\n    TAR_MAT: 4\n')
assert blk in t, "upstream deck no longer carries the structure -> scatra2 clone"
if sys.argv[3] == "yes":
    t = t.replace(blk, "")
open(sys.argv[2], "w").write(t)
PY
python3 - "$BASE" "$TMP/noale.yaml" "$DROP_ALE_CLONE" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = ('  - SRC_FIELD: "fluid"\n    SRC_MAT: 1\n'
       '    TAR_FIELD: "ale"\n    TAR_MAT: 10\n')
assert blk in t, "upstream deck no longer carries the fluid -> ale clone"
if sys.argv[3] == "yes":
    t = t.replace(blk, "")
open(sys.argv[2], "w").write(t)
PY
echo "FULL_CLONE_ENTRIES=$(grep -c 'SRC_FIELD:' "$TMP/full.yaml")"
echo "NOSCATRA2_CLONE_ENTRIES=$(grep -c 'SRC_FIELD:' "$TMP/noscatra2.yaml")"
echo "NOALE_CLONE_ENTRIES=$(grep -c 'SRC_FIELD:' "$TMP/noale.yaml")"

probe FULL       "$TMP/full.yaml"
probe NOSCATRA2  "$TMP/noscatra2.yaml"
probe NOALECLONE "$TMP/noale.yaml"

grep -m1 -F "OK (3)" "$TMP/FULL.log"
grep -m1 -F "processor 0 finished normally" "$TMP/FULL.log"
grep -m1 -F "terminate called after throwing an instance of 'std::out_of_range'" "$TMP/NOSCATRA2.log"
grep -m1 -F "terminate called after throwing an instance of 'std::out_of_range'" "$TMP/NOALECLONE.log"

# 4C never gets to say anything.
echo "NOSCATRA2_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/NOSCATRA2.log")"
echo "NOALECLONE_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/NOALECLONE.log")"
echo "CLAIMED_CANNOT_CLONE_TEXT=$(grep -ci 'cannot clone material' "$TMP/NOSCATRA2.log")$(grep -ci 'cannot clone material' "$TMP/NOALECLONE.log")"
echo "NOSCATRA2_MENTIONS_CLONING=$(grep -ci 'cloning' "$TMP/NOSCATRA2.log")"
echo "NOSCATRA2_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/NOSCATRA2.log")"
exit 0
