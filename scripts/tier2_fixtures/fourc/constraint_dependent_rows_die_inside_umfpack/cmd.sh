#!/bin/bash
# Tier-2 for fourc::constraint#0 — constraint equations do have to be linearly
# independent, and duplicating one is fatal. But 4C does not tell you so.
#
# Claimed:  "direct LU reports 'zero pivot in Schur complement'".
# Observed: no such message exists in 4C or in Trilinos, and none is printed.
#           The duplicated row makes the saddle-point system singular, 4C hands
#           it to UMFPACK through Constraints::ConstraintSolver::solve_direct,
#           and the back-substitution divides by the zero pivot:
#
#     Signal: Floating point exception (8)
#     Signal code: Invalid floating point operation (7)
#     libumfpack.so.5(umfdi_usolve+...)
#
#   Exit status 136, zero "PROC 0 ERROR" lines, no mention of constraints or
#   rank. A caller that only greps for a 4C error message sees a clean log and
#   a dead process.
#
# Upstream deck: constr2D_MPC_dist — a 2D wall with one DESIGN LINE MULTIPNT
# CONSTRAINT 2D over nodes 1/2/3, solved with NLNSOL newtonlinuzawa. The bad arm
# adds a second condition, ConditionID 2, over exactly the same three nodes.
. "$(dirname "$0")/../_lib/preamble.sh"
ulimit -c 0   # the bad arm dies of SIGFPE; do not leave core files behind

BASE=$(upstream constr2D_MPC_dist.4C.yaml) || exit 3
grep -q "DESIGN LINE MULTIPNT CONSTRAINT 2D" "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$BASE" "$TMP/independent.yaml"

python3 - "$BASE" "$TMP/dependent.yaml" <<'PY'
import sys
t = open(sys.argv[1]).read()
one = """DESIGN LINE MULTIPNT CONSTRAINT 2D:
  - E: 1
    ConditionID: 1
    amplitude: 1
    curve: 1
    constrNode1: 1
    constrNode2: 2
    constrNode3: 3
    activeTime: 0
"""
assert one in t, "upstream deck no longer carries the single 1/2/3 MPC"
twice = one + """  - E: 1
    ConditionID: 2
    amplitude: 1
    curve: 1
    constrNode1: 1
    constrNode2: 2
    constrNode3: 3
    activeTime: 0
"""
open(sys.argv[2], "w").write(t.replace(one, twice, 1))
PY

probe INDEPENDENT "$TMP/independent.yaml"
probe DEPENDENT   "$TMP/dependent.yaml"

# Control: one constraint, ten steps, result test passes.
grep -m1 -F "processor 0 finished normally" "$TMP/INDEPENDENT.log"
echo "INDEPENDENT_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/INDEPENDENT.log")"

# Duplicate the row and the process is killed by the FPU, inside UMFPACK,
# reached through 4C's own constraint solver.
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/DEPENDENT.log"
grep -m1 -F "umfdi_usolve" "$TMP/DEPENDENT.log"
grep -m1 -F "ConstraintSolver" "$TMP/DEPENDENT.log"
grep -m1 -F "uzawa_linear_newton_full" "$TMP/DEPENDENT.log"

python3 - "$TMP/DEPENDENT.log" <<'PY'
import sys
log = open(sys.argv[1], "rb").read().decode("utf-8", "replace").lower()
print("DEPENDENT_4C_DIAGNOSTICS=%d" % log.count("proc 0 error"))
print("CLAIMED_ZERO_PIVOT_TEXT=%d" % log.count("zero pivot"))
print("DEPENDENT_MENTIONS_RANK_OR_SINGULAR=%d"
      % (log.count("rank-deficient") + log.count("singular")))
PY
exit 0
