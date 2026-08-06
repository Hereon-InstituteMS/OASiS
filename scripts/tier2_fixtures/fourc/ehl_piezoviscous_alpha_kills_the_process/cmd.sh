#!/bin/bash
# Tier-2 for fourc::ehl#2 — a too-large pressure-viscosity coefficient does not
# "diverge or oscillate between two states".  It takes the whole process down
# with SIGFPE, and the relaxation the entry recommends cannot help because those
# knobs are inert under 4C's default EHL coupling algorithm.
#
# Upstream ehl_bearing_barus.4C.yaml uses MAT_lubrication_law_barus with
# PreVisCoeff (= alpha) 0.0112.  Multiply it by 100:
#
#   numiter   abs-res-norm   abs-inc-norm ...
#         1    5.33839e+01    9.92739e+00
#         2    7.95637e+70    4.35138e+43
#   Signal: Floating point exception (8)
#   Signal code: Floating point divide-by-zero (3)
#   ... LubricationEleCalc<...>::calc_mat_psl ...
#
# One Newton step from 5e1 to 8e70, then a divide-by-zero inside the lubrication
# element.  The shell reports 136 (128 + SIGFPE); there is no 4C error message,
# no "did not converge", and no result-test verdict — MPI_Abort is never reached,
# so a caller that only greps for 4C diagnostics sees NOTHING.
#
# The entry's remedy is also wrong for the default setup: MAXOMEGA / MINOMEGA /
# STARTOMEGA live in ELASTO HYDRO DYNAMIC/PARTITIONED, which is only consulted
# when ELASTO HYDRO DYNAMIC/COUPALGO is ehl_IterStagg.  The default is
# ehl_Monolithic, so adding them changes nothing at all: the RELAX arm below
# dies exactly the same way.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream ehl_bearing_barus.4C.yaml) || exit 3
grep -q '      PreVisCoeff: 0.0112' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_previscoeff_changed"; exit 3; }
grep -q '^ELASTO HYDRO DYNAMIC/MONOLITHIC:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_monolithic_section_changed"; exit 3; }
grep -q 'COUPALGO' "$BASE" \
  && { echo "FIXTURE_ABORT=upstream_now_sets_coupalgo"; exit 3; }

# The pathology: the pressure-viscosity coefficient the bad arms are given.
BIG_PREVISCOEFF=1.12
RELAX_BLOCK='ELASTO HYDRO DYNAMIC/PARTITIONED:\n  STARTOMEGA: 0.3\n  MAXOMEGA: 0.5\n  MINOMEGA: 0.1\n'

cp "$BASE" "$TMP/tuned.yaml"
sed "s/      PreVisCoeff: 0.0112/      PreVisCoeff: $BIG_PREVISCOEFF/" "$BASE" > "$TMP/bigalpha.yaml"
python3 - "$TMP/bigalpha.yaml" "$TMP/relax.yaml" "$RELAX_BLOCK" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = sys.argv[3].encode().decode("unicode_escape")
assert "ELASTO HYDRO DYNAMIC/MONOLITHIC:\n" in t
open(sys.argv[2], "w").write(
    t.replace("ELASTO HYDRO DYNAMIC/MONOLITHIC:\n",
              blk + "ELASTO HYDRO DYNAMIC/MONOLITHIC:\n", 1))
PY
grep -m1 '      PreVisCoeff:' "$TMP/bigalpha.yaml" | tr -d ' ' | sed 's/^/BAD_ARM_/'
echo "RELAX_ARM_HAS_OMEGA_KEYS=$(grep -c 'STARTOMEGA' "$TMP/relax.yaml")"

probe TUNED    "$TMP/tuned.yaml"
probe BIGALPHA "$TMP/bigalpha.yaml"
probe RELAX    "$TMP/relax.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/TUNED.log"
grep -m1 -F "OK (4)" "$TMP/TUNED.log"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/BIGALPHA.log"
grep -m1 -F "Signal code: Floating point divide-by-zero (3)" "$TMP/BIGALPHA.log"
grep -m1 -oF "LubricationEleCalc" "$TMP/BIGALPHA.log"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/RELAX.log"

# There is no 4C-side diagnostic of any kind: the process is killed mid-assembly.
echo "BIGALPHA_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/BIGALPHA.log")"
echo "BIGALPHA_NONCONVERGENCE_MESSAGES=$(grep -ci 'did not converge' "$TMP/BIGALPHA.log")"
echo "BIGALPHA_RESULT_TEST_LINES=$(grep -c 'Checking results of' "$TMP/BIGALPHA.log")"
# The recommended relaxation knobs are accepted and inert.
echo "RELAX_ARM_STILL_DIES=$(grep -c 'Signal: Floating point exception' "$TMP/RELAX.log")"
echo "RELAX_IGNORED_WARNINGS=$(grep -ciE 'omega.*(ignor|unus|no effect)|partitioned.*ignor' "$TMP/RELAX.log")"
exit 0
