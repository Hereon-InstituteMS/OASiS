#!/bin/bash
# Tier-2 for fourc::ssi#5 — and a FALSIFICATION of both halves of its Signal.
#
# Claimed: "'sparse' with a Belos GMRES + block-Teko preconditioner ignores the
#          block structure — preconditioner iterations explode; 'block' with a
#          direct UMFPACK solver wastes memory constructing block sub-blocks."
#
# Observed: neither is a performance story.  Both mismatches are hard aborts,
# and the harmless case the entry warned about is genuinely harmless.
#
#   SPARSE_DIRECT  MATRIXTYPE sparse, UMFPACK           -> runs (the baseline)
#   BLOCK_DIRECT   MATRIXTYPE block,   UMFPACK          -> runs, all tests pass.
#                  No waste, no warning, no penalty at all.
#   BLOCKSCATRA    scatra MATRIXTYPE block_condition with a direct solver ->
#                  "Global system matrix with block structure requires AMGnxn,
#                  MueLu or Teko block preconditioner!"
#                  (src/scatra/4C_scatra_timint_implicit.cpp)
#   TEKO_SPARSE    upstream Teko/BGS-AMG deck with SSI MATRIXTYPE forced to
#                  sparse -> "Incompatible matrix type associated with scalar
#                  transport field!"  (src/ssi/4C_ssi_monolithic.cpp)
#
# So the thing to match is not "MATRIXTYPE to the SOLVER type" in the abstract:
# it is SSI CONTROL/MONOLITHIC's MATRIXTYPE to SCALAR TRANSPORT DYNAMIC's
# MATRIXTYPE, and the scatra matrix type to the preconditioner.  Both couplings
# are checked and both abort.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssi_mono_3D_1hex8_scatra.4C.yaml) || exit 3
TEKO=$(upstream ssi_mono_3D_hex8_elch_s2i_butlervolmer_grain_boundary_meshtying_BGS-AMG_5x5.4C.yaml) || exit 3
ln -s "$(dirname "$TEKO")/xml" "$TMP/xml"
grep -q '  MATRIXTYPE: "sparse"' "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q '  MATRIXTYPE: "block"'  "$TEKO" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "TEKO_XML_FILE"          "$TEKO" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/sparse_direct.yaml"
sed 's/  MATRIXTYPE: "sparse"/  MATRIXTYPE: "block"/' "$BASE" > "$TMP/block_direct.yaml"
python3 - "$BASE" "$TMP/blockscatra.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
head = 'SCALAR TRANSPORT DYNAMIC:\n  SOLVERTYPE: "nonlinear"\n'
if head not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(
    t.replace(head, head + '  MATRIXTYPE: "block_condition"\n', 1))
PY
[ -f "$TMP/blockscatra.yaml" ] || exit 3
sed 's/^  MATRIXTYPE: "block"$/  MATRIXTYPE: "sparse"/' "$TEKO" > "$TMP/teko_sparse.yaml"

probe SPARSEDIRECT "$TMP/sparse_direct.yaml"
probe BLOCKDIRECT  "$TMP/block_direct.yaml"
probe BLOCKSCATRA  "$TMP/blockscatra.yaml"
probe TEKOSPARSE   "$TMP/teko_sparse.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/SPARSEDIRECT.log"
# 'block' with a direct solver is not punished in any way.
echo "BLOCKDIRECT_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/BLOCKDIRECT.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BLOCKDIRECT.log"
# A block scatra matrix with a direct solver is refused, and the message names
# the preconditioners that would have been acceptable.
grep -m1 -F "Global system matrix with block structure requires AMGnxn, MueLu or Teko block preconditioner!" "$TMP/BLOCKSCATRA.log"
grep -m1 -oF "4C_scatra_timint_implicit.cpp" "$TMP/BLOCKSCATRA.log"
# And a sparse SSI matrix over a block scatra field is refused too.
grep -m1 -F "Incompatible matrix type associated with scalar transport field!" "$TMP/TEKOSPARSE.log"
grep -m1 -oF "4C_ssi_monolithic.cpp" "$TMP/TEKOSPARSE.log"
echo "TEKOSPARSE_FAILS_IN_SETUP_SYSTEM=$(grep -c 'SsiMono::setup_system' "$TMP/TEKOSPARSE.log")"
# Neither mismatch degrades into slow convergence: neither run reaches a solve.
echo "TEKOSPARSE_RESULT_TESTS_PERFORMED=$(grep -c 'is WRONG --> actresult=' "$TMP/TEKOSPARSE.log")"
exit 0
