#!/bin/bash
# Tier-2 for fourc::lubrication#4 — deleting the pressure Dirichlet from a
# Reynolds deck does NOT make the solver report a zero pivot.  It solves, and
# hands back a badly wrong pressure with no diagnostic at all.
#
# Claimed:  "a pure-Neumann setup makes the Reynolds operator singular ...
#            direct LU solver reports 'zero pivot'".
# Observed: upstream lubrication_sb_2d.4C.yaml with its DESIGN LINE DIRICH
#           CONDITIONS block deleted runs all five steps on Superlu, converges,
#           and returns 761.33 at node 10 against a reference of 73.94 — an
#           order of magnitude out.  The strings 'zero pivot' and 'singular'
#           appear nowhere; the only complaint is the deck's own result test.
#           With the cavitation penalty also switched off, the pure-Neumann
#           problem still solves (to ~-3.6e-07) and still says nothing.
#
# So the real hazard is the opposite of what was written: the failure is SILENT,
# and an input without a pressure Dirichlet must be caught by the author, not by
# the linear solver.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream lubrication_sb_2d.4C.yaml) || exit 3
grep -q 'DESIGN LINE DIRICH CONDITIONS:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_dirich_section_changed"; exit 3; }
grep -q 'SOLVER: "Superlu"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_solver_changed"; exit 3; }

DROP_DIRICHLET=yes

cp "$BASE" "$TMP/with_dbc.yaml"
python3 - "$BASE" "$TMP/no_dbc.yaml" "$DROP_DIRICHLET" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
if sys.argv[3] == "yes":
    t2 = re.sub(r'DESIGN LINE DIRICH CONDITIONS:\n(  - E: 1\n(    .*\n)+)', '', t)
    assert 'DESIGN LINE DIRICH' not in t2, "Dirichlet block not removed"
    t = t2
open(sys.argv[2], "w").write(t)
PY
# also drop the cavitation penalty, so nothing regularises the operator
sed 's/PENALTY_CAVITATION: 1e+08/PENALTY_CAVITATION: 0.0/' "$TMP/no_dbc.yaml" \
    > "$TMP/no_dbc_no_pen.yaml"

probe WITH_DBC      "$TMP/with_dbc.yaml"
probe NO_DBC        "$TMP/no_dbc.yaml"
probe NO_DBC_NO_PEN "$TMP/no_dbc_no_pen.yaml"

grep -m1 -F "is CORRECT, abs(diff)=" "$TMP/WITH_DBC.log"
grep -m1 -F "processor 0 finished normally" "$TMP/WITH_DBC.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/NO_DBC.log"
grep -m1 -F "Result check failed with 1 errors out of 1 tests" "$TMP/NO_DBC.log"

# It reached and completed the last time step.
echo "NO_DBC_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/NO_DBC.log")"
echo "NO_DBC_NEWTON_LINES=$(grep -c '\[L_2 \]' "$TMP/NO_DBC.log")"
# The claimed solver diagnostics are absent, with or without the penalty.
echo "CLAIMED_ZERO_PIVOT=$(grep -ci 'zero pivot' "$TMP/NO_DBC.log")$(grep -ci 'zero pivot' "$TMP/NO_DBC_NO_PEN.log")"
echo "CLAIMED_SINGULAR=$(grep -ci 'singular' "$TMP/NO_DBC.log")$(grep -ci 'singular' "$TMP/NO_DBC_NO_PEN.log")"
echo "NO_DBC_MISSING_DIRICHLET_WARNINGS=$(grep -ciE 'dirichlet.*(missing|absent|none)|no dirichlet|not well.?posed|ill.?posed' "$TMP/NO_DBC.log")"
P=$(grep -m1 -oE 'actresult=[ ]*[-0-9.eE+]+' "$TMP/NO_DBC.log" | tr -d ' ' | cut -d= -f2)
echo "NO_DBC_PRESSURE=$P"
echo "NO_DBC_PRESSURE_IS_FINITE=$(python3 -c "import math;print('yes' if math.isfinite($P) else 'no')")"
echo "NO_DBC_OVER_REFERENCE=$(python3 -c "print('%.2f' % ($P/73.94356207110268))")"
exit 0
