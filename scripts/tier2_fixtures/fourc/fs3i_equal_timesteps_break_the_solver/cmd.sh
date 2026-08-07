#!/bin/bash
# Tier-2 for fourc::fs3i#6 — "Set TIMESTEP equal in both fields" is the OPPOSITE
# of what the working reference does, and following it breaks the run.
#
# Upstream fs3i_part_1wc_infperm.4C.yaml is a passing 4C regression case and it
# deliberately does NOT use one time step everywhere:
#     FS3I DYNAMIC             TIMESTEP 12.5   <- the outer coupling step
#     FSI DYNAMIC              TIMESTEP 25
#     SCALAR TRANSPORT DYNAMIC TIMESTEP 25
# The FS3I step is half the field step; that is the deck that passes its three
# result tests.
#
# Set FS3I DYNAMIC/TIMESTEP to 25 so all three agree — exactly the entry's advice
# — and the run does not merely drift: it fails outright with
#   "Core::LinearSolver::BelosSolver: Iterative solver did not converge."
# from core/linear_solver/src/method/4C_linear_solver_method_iterative.cpp line
# 193.  The entry's "result drifts from a fully-coupled monolithic reference by
# 5-15%" describes nothing that happens here.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fs3i_part_1wc_infperm.4C.yaml) || exit 3
grep -q 'FS3I DYNAMIC:' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fs3i_dynamic_changed"; exit 3; }
grep -q '  TIMESTEP: 12.5' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_fs3i_timestep_changed"; exit 3; }
cp -r "$(dirname "$BASE")/xml" "$TMP/" \
  || { echo "FIXTURE_ABORT=missing_teko_xml"; exit 3; }
cd "$TMP" || exit 3

# The pathology: the outer FS3I coupling step, made equal to the field step.
FS3I_TIMESTEP=25

cp "$BASE" "$TMP/staggered.yaml"
python3 - "$BASE" "$TMP/equal.yaml" "$FS3I_TIMESTEP" <<'PY'
import sys
t = open(sys.argv[1]).read()
old = "FS3I DYNAMIC:\n  TIMESTEP: 12.5"
assert old in t, "upstream deck no longer sets the FS3I coupling step to 12.5"
open(sys.argv[2], "w").write(
    t.replace(old, "FS3I DYNAMIC:\n  TIMESTEP: %s" % sys.argv[3], 1))
PY
python3 - "$TMP/equal.yaml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
fs3i = re.search(r'FS3I DYNAMIC:\n  TIMESTEP: ([\d.]+)', t).group(1)
fsi = re.search(r'FSI DYNAMIC:\n(?:  \S.*\n)*?  TIMESTEP: ([\d.]+)', t).group(1)
print("EQUAL_ARM_[FS3I_DT:%s;FSI_DT:%s]" % (fs3i, fsi))
print("EQUAL_ARM_STEPS_MATCH=%s" % ("yes" if float(fs3i) == float(fsi) else "no"))
PY

probe STAGGERED "$TMP/staggered.yaml"
probe EQUAL     "$TMP/equal.yaml"

grep -m1 -F "OK (3)" "$TMP/STAGGERED.log"
grep -m1 -F "processor 0 finished normally" "$TMP/STAGGERED.log"
grep -m1 -F "Core::LinearSolver::BelosSolver: Iterative solver did not converge." "$TMP/EQUAL.log"
grep -m1 -F "4C_linear_solver_method_iterative.cpp" "$TMP/EQUAL.log"

echo "STAGGERED_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/STAGGERED.log")"
echo "EQUAL_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/EQUAL.log")"
echo "EQUAL_NONCONVERGENCE=$(grep -c 'did not converge' "$TMP/EQUAL.log")"
# There is no percentage drift to compare: the equal-step run produces no result.
echo "EQUAL_RESULT_LINES=$(grep -c 'actresult=' "$TMP/EQUAL.log")"
exit 0
