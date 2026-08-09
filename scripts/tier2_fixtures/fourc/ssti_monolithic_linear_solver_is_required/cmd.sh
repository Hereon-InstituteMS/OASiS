#!/bin/bash
# Tier-2 for fourc::ssti#4 — SSTI CONTROL/MONOLITHIC's LINEAR_SOLVER is
# required, and forgetting it produces one of the least helpful failures in 4C.
#
# LINEAR_SOLVER defaults to -1, and nothing checks that.  4C goes looking for a
# section literally called "SOLVER -1", does not find it, and dies inside
# Teuchos rather than in 4C:
#
#     terminate called after throwing an instance of
#     'Teuchos::Exceptions::InvalidParameterName'
#     ... in the parameter (sub)list "ROOT->SOLVER -1".
#
# — SIGABRT, exit 134, no "PROC 0 ERROR" banner, no source file named, and no
# mention of SSTI or of LINEAR_SOLVER.  The only clue is the "-1".  All of that
# is asserted, because it is what an agent has to recognise.
#
# The entry's other half — that UMFPACK "works for small problems but is
# memory-prohibitive" — is a sizing opinion, not a 4C behaviour; the upstream
# deck points LINEAR_SOLVER at a direct solver and passes, which is asserted as
# the baseline.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ssti_mono_3D_3hex8_elch_s2i_butlervolmerthermo_growthlaw.4C.yaml) || exit 3
grep -q "^SSTI CONTROL/MONOLITHIC:" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$BASE" "$TMP/withsolver.yaml"

python3 - "$BASE" "$TMP/nosolver.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = 'SSTI CONTROL/MONOLITHIC:\n  ABSTOLRES: 1e-11\n  LINEAR_SOLVER: 1\n'
if blk not in t:
    print("FIXTURE_ABORT=upstream_deck_changed"); sys.exit(3)
open(sys.argv[2], "w").write(
    t.replace(blk, 'SSTI CONTROL/MONOLITHIC:\n  ABSTOLRES: 1e-11\n', 1))
PY
[ -f "$TMP/nosolver.yaml" ] || exit 3

probe WITHSOLVER "$TMP/withsolver.yaml"
probe NOSOLVER   "$TMP/nosolver.yaml"

# The upstream deck reaches a direct solver and passes every result test.
grep -m1 -F "processor 0 finished normally" "$TMP/WITHSOLVER.log"
echo "WITHSOLVER_RESULT_TEST_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/WITHSOLVER.log")"
# Omitting it: a Teuchos exception naming the defaulted id, not a 4C message.
grep -m1 -F "Teuchos::Exceptions::InvalidParameterName" "$TMP/NOSOLVER.log"
grep -m1 -F 'in the parameter (sub)list "ROOT->SOLVER -1".' "$TMP/NOSOLVER.log"
# None of 4C's usual signposts are present.
echo "NOSOLVER_HAS_4C_ERROR_BANNER=$(grep -c 'PROC 0 ERROR in' "$TMP/NOSOLVER.log")"
echo "NOSOLVER_NAMES_LINEAR_SOLVER=$(grep -c 'LINEAR_SOLVER' "$TMP/NOSOLVER.log")"
# SSTI is named only inside mangled backtrace symbols (_ZN5FourC4SSTI...),
# never in a readable message.
echo "NOSOLVER_NAMES_SSTI_IN_PROSE=$(grep 'SSTI' "$TMP/NOSOLVER.log" | grep -vc '_ZN5FourC4SSTI')"
exit 0
