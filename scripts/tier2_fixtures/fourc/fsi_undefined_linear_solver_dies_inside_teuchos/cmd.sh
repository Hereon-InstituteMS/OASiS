#!/bin/bash
# Tier-2 for fourc::fsi#5 — both halves of the entry, executed.
#
# (a) "reusing one SOLVER for all three fields is ALLOWED": upstream
#     fsi_fp_mono_fs_ga_ga.4C.yaml defines exactly ONE solver block, SOLVER 1,
#     and points STRUCTURAL DYNAMIC, FLUID DYNAMIC, ALE DYNAMIC and
#     FSI DYNAMIC/MONOLITHIC SOLVER all at it.  It runs and matches all six
#     pinned results.  So the "own SOLVER N per field" advice is a preference,
#     and 4C never comments on the sharing.
#
# (b) "referencing a SOLVER that is not defined raises 'SOLVER N not found' at
#     setup": FALSE.  Pointing ALE DYNAMIC at LINEAR_SOLVER: 7 with no SOLVER 7
#     block does not produce a 4C diagnostic at all.  4C hands the empty
#     sublist to Trilinos, which throws
#       Teuchos::Exceptions::InvalidParameterName
#       Error!  The parameter "SOLVER" does not exist
#       in the parameter (sub)list "ROOT->SOLVER 7".
#     uncaught.  std::terminate runs, the process dies on SIGABRT (exit 134),
#     there is no "PROC 0 ERROR" block, no MPI_ABORT banner and no mention of
#     ALE DYNAMIC — the only clue to the offending section is the mangled
#     Adapter::AleBaseAlgorithm::setup_ale frame in the signal backtrace.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_fp_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q '^SOLVER 1:' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_has_no_solver_1"; exit 3; }
echo "UPSTREAM_SOLVER_BLOCKS=$(grep -c '^SOLVER [0-9]*:' "$BASE")"
echo "UPSTREAM_FIELDS_POINTING_AT_A_SOLVER=$(grep -c '^  LINEAR_SOLVER: 1$' "$BASE")"

# The pathology: point one field at a SOLVER block that does not exist.
ALE_LINEAR_SOLVER=7

cp "$BASE" "$TMP/shared.yaml"
python3 - "$BASE" "$TMP/dangling.yaml" "$ALE_LINEAR_SOLVER" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = "ALE DYNAMIC:\n  ALE_TYPE: springs_spatial\n  LINEAR_SOLVER: 1"
assert blk in t, "upstream ALE DYNAMIC block changed"
t = t.replace(blk, blk[:-1] + sys.argv[3], 1)
open(sys.argv[2], "w").write(t)
PY
echo "DANGLING_ALE_SOLVER_ID=$(grep -A2 '^ALE DYNAMIC:' "$TMP/dangling.yaml" | grep -o 'LINEAR_SOLVER: [0-9]*' | grep -o '[0-9]*')"
echo "DANGLING_HAS_THAT_SOLVER_BLOCK=$(grep -c "^SOLVER $ALE_LINEAR_SOLVER:" "$TMP/dangling.yaml")"

probe SHARED   "$TMP/shared.yaml"
probe DANGLING "$TMP/dangling.yaml"

# (a) one shared solver for all four consumers is fine.
grep -m1 -F "processor 0 finished normally" "$TMP/SHARED.log"
grep -m1 -F "OK (6)" "$TMP/SHARED.log"
echo "SHARED_SOLVER_WARNINGS=$(grep -ciE 'solver.*(shared|reused|suboptimal)' "$TMP/SHARED.log")"

# (b) the dangling reference is not a 4C diagnostic at all.
grep -m1 -F 'terminate called after throwing an instance of' "$TMP/DANGLING.log"
grep -m1 -F 'Teuchos::Exceptions::InvalidParameterName' "$TMP/DANGLING.log"
grep -m1 -F 'The parameter "SOLVER" does not exist' "$TMP/DANGLING.log"
grep -m1 -F 'in the parameter (sub)list "ROOT->SOLVER 7".' "$TMP/DANGLING.log"
grep -m1 -F 'AleBaseAlgorithm9setup_ale' "$TMP/DANGLING.log"

echo "DANGLING_HAS_PROC0_ERROR_BLOCK=$(grep -c 'PROC 0 ERROR' "$TMP/DANGLING.log")"
echo "DANGLING_HAS_MPI_ABORT_BANNER=$(grep -c 'MPI_ABORT was invoked' "$TMP/DANGLING.log")"
echo "DANGLING_CLAIMED_TEXT=$(grep -ciE 'SOLVER [0-9]+ not found' "$TMP/DANGLING.log")"
echo "DANGLING_NAMES_ALE_DYNAMIC=$(grep -c 'ALE DYNAMIC' "$TMP/DANGLING.log")"
exit 0
