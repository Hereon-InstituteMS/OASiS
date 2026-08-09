#!/bin/bash
# Tier-2 for fourc::multiscale#0 — the cost is real and the profile will not tell
# you where it went.
#
# Claimed: "profile log shows >95% of time in MicroSolver::Solve".
# Observed: there is no MicroSolver timer. The string does not appear once in the
#          run's output. Every RVE solve is counted into the SAME generic
#          Core::LinAlg::Solver: 2) Solve entry as the macro solves, so the
#          TimeMonitor table cannot attribute the time at all.
#
# What IS visible, and is the honest way to see the cost, is the call count.
# Upstream sohex8_multiscale_macro on a tiny macro mesh performs 1.326e+04 linear
# solves. Swap the three MAT_Struct_Multiscale materials for a plain
# MAT_Struct_StVenantKirchhoff on the identical mesh and schedule and it performs
# 10. Same macro problem, three orders of magnitude more linear algebra, and
# nothing in the profile says the word micro.
#
# Both arms have their RESULT DESCRIPTION stripped so both reach the TimeMonitor
# table — 4C aborts before printing it when a result test fails, which would
# otherwise hide the comparison.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream sohex8_multiscale_macro.4C.yaml) || exit 3
MICRO=$(upstream sohex8_multiscale_micro.mat.4C.yaml) || exit 3
cd "$TMP" || exit 3
cp "$BASE" macro.yaml
grep -q "    MAT_Struct_Multiscale:" macro.yaml || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

mkdir -p d_ms d_plain
cp "$MICRO" d_ms/; cp "$MICRO" d_plain/

python3 - macro.yaml d_ms/macro.yaml d_plain/macro.yaml <<'PY'
import re, sys
t = open(sys.argv[1]).read()

def strip_result_description(s):
    i = s.find("RESULT DESCRIPTION:")
    if i < 0:
        return s
    m = re.search(r"\n(?=[A-Z][A-Z0-9 /_-]*:\n)", s[i + 1:])
    return s[:i] + (s[i + 1 + m.start() + 1:] if m else "")

plain = re.sub(r"    MAT_Struct_Multiscale:\n(?:      \w+: [^\n]*\n)+",
               "    MAT_Struct_StVenantKirchhoff:\n"
               "      YOUNG: 100.0\n      NUE: 0.3\n      DENS: 1.0\n", t)
assert "MAT_Struct_Multiscale" not in plain, "material substitution failed"
open(sys.argv[2], "w").write(strip_result_description(t))
open(sys.argv[3], "w").write(strip_result_description(plain))
PY

( cd d_ms    && stdbuf -oL -eL "$BIN" macro.yaml res > run.log 2>&1; echo "EXIT_MULTISCALE=$?" )
( cd d_plain && stdbuf -oL -eL "$BIN" macro.yaml res > run.log 2>&1; echo "EXIT_PLAIN=$?" )

grep -m1 -F "processor 0 finished normally" d_ms/run.log
calls() { grep -m1 "Core::LinAlg::Solver:  2)   Solve" "$1" | sed 's/.*(\(.*\)).*/\1/'; }
echo "MULTISCALE_SOLVE_CALLS=$(calls d_ms/run.log)"
echo "PLAIN_SOLVE_CALLS=$(calls d_plain/run.log)"
# No timer anywhere names the micro scale.
echo "MICROSOLVER_TIMERS=$(grep -ci 'microsolver' d_ms/run.log)"
echo "TIMERS_MENTIONING_MICRO=$(sed -n '/TimeMonitor results/,/Total wall time/p' d_ms/run.log | grep -ci micro)"
exit 0
