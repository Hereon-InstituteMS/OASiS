#!/bin/bash
# Tier-2 for fourc::fpsi#0 — FPSI does need ALE DYNAMIC and a fluid -> ale
# cloning entry, but neither of the two quoted diagnostics exists, and the two
# failures are of very different quality.
#
# Claimed:  missing ALE DYNAMIC raises 'FPSI: ALE field not configured';
#           missing CLONING MATERIAL MAP raises 'cannot clone material for ALE'.
# Observed, on upstream fpsi_ofsiinterface.4C.yaml:
#   NOALE   : "No linear solver defined for ALE problems. Please set
#              LINEAR_SOLVER in ALE DYNAMIC to a valid number!" from
#              adapter/4C_adapter_ale.cpp line 89.  A clean, actionable 4C abort
#              — it just talks about the solver, not about FPSI or the field.
#   NOCLONE : no 4C diagnostic at all.  Deleting the fluid -> ale entry from
#              CLONING MATERIAL MAP throws an uncaught std::out_of_range:
#              "terminate called after throwing an instance of 'std::out_of_range'"
#              and the process is SIGABRTed (shell status 134) in the middle of
#              boundary_conditions_geometry().  Nothing names the map, the
#              field, or the material.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fpsi_ofsiinterface.4C.yaml) || exit 3
grep -q '^ALE DYNAMIC:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_ale_section_changed"; exit 3; }
grep -q '^CLONING MATERIAL MAP:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_cloning_map_changed"; exit 3; }

# The two pathologies.
DROP_ALE_SECTION=yes
DROP_FLUID_TO_ALE_CLONE=yes

cp "$BASE" "$TMP/full.yaml"
python3 - "$BASE" "$TMP/noale.yaml" "$DROP_ALE_SECTION" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
if sys.argv[3] == "yes":
    t2 = re.sub(r'ALE DYNAMIC:\n(  \S.*\n)+', '', t, count=1)
    assert 'ALE DYNAMIC:' not in t2, "ALE DYNAMIC block not removed"
    t = t2
open(sys.argv[2], "w").write(t)
PY
python3 - "$BASE" "$TMP/noclone.yaml" "$DROP_FLUID_TO_ALE_CLONE" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = ('  - SRC_FIELD: "fluid"\n    SRC_MAT: 4\n'
       '    TAR_FIELD: "ale"\n    TAR_MAT: 5\n')
assert blk in t, "upstream deck no longer carries the fluid -> ale clone entry"
if sys.argv[3] == "yes":
    t = t.replace(blk, "")
open(sys.argv[2], "w").write(t)
PY
echo "NOALE_DECK_HAS_ALE_SECTION=$(grep -c '^ALE DYNAMIC:' "$TMP/noale.yaml")"
echo "NOCLONE_DECK_HAS_ALE_CLONE=$(grep -c 'TAR_FIELD: "ale"' "$TMP/noclone.yaml")"

probe FULL    "$TMP/full.yaml"
probe NOALE   "$TMP/noale.yaml"
probe NOCLONE "$TMP/noclone.yaml"

grep -m1 -F "OK (2)" "$TMP/FULL.log"
grep -m1 -F "processor 0 finished normally" "$TMP/FULL.log"
grep -m1 -F "No linear solver defined for ALE problems. Please set LINEAR_SOLVER in ALE DYNAMIC to a valid number!" "$TMP/NOALE.log"
grep -m1 -F "4C_adapter_ale.cpp" "$TMP/NOALE.log"
grep -m1 -F "terminate called after throwing an instance of 'std::out_of_range'" "$TMP/NOCLONE.log"

# Neither quoted string exists.
echo "CLAIMED_ALE_FIELD_TEXT=$(grep -ci 'ALE field not configured' "$TMP/NOALE.log")"
echo "CLAIMED_CANNOT_CLONE_TEXT=$(grep -ci 'cannot clone material' "$TMP/NOCLONE.log")"
# The cloning failure gives 4C no chance to say anything.
echo "NOCLONE_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/NOCLONE.log")"
echo "NOCLONE_MENTIONS_CLONING=$(grep -ci 'cloning' "$TMP/NOCLONE.log")"
exit 0
