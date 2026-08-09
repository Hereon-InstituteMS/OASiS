#!/bin/bash
# Tier-2 for fourc::fsi#20 — a step inflow does not diverge.  It converges, in
# 83% more Newton iterations, at a different answer.
#
# Claimed: "a fast inflow ramp (e.g. step or 1s rise) over a flexible structure
#           produces Newton divergence within ~10 time steps even at laminar Re
#           — the structural response cannot follow the fluid forcing transient.
#           For initial testing, use a slow ramp (5-10s period, e.g. cos(pi*t/5))."
# Observed: the upstream 2D FSI driven-cavity benchmark
#           fsi_dc_mono_fs_ga_ga.4C.yaml already uses exactly the recommended
#           slow ramp — FUNCT1 = y*(1-cos(2*t*pi/5)), FUNCT2 = (1-cos(2*t*pi/5)),
#           a 5 s cosine — over 10 steps of dt = 0.1.  Replacing it with an
#           INSTANTANEOUS step to the same amplitude (y*2 and 2, the peak the
#           cosine would reach) keeps the amplitude fixed and changes only the
#           rate.  The result: both runs complete all 10 steps, neither prints
#           any non-convergence message, and the step run takes 148 Newton steps
#           against 81.  It exits nonzero only because the answer moved — all 9
#           pinned results, e.g. structure dispx at node 38 = -2.650e-03 against
#           the pinned 2.467e-05.
#
#           So the cost of a fast ramp here is iterations and a different
#           solution, not divergence; "Newton divergence within ~10 time steps"
#           is not what a step forcing does to this benchmark.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fsi_dc_mono_fs_ga_ga.4C.yaml) || exit 3
grep -q 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "y\*(1-cos(2\*t\*pi/5))"' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_deck_ramp_changed"; exit 3; }

# The pathology: replace the 5 s cosine ramp with an instantaneous step to the
# same peak amplitude.
FAST_RAMP_SPACE_TIME="y*2"
FAST_RAMP_TIME="2"

python3 - "$BASE" "$TMP" "$DECKS" "$FAST_RAMP_SPACE_TIME" "$FAST_RAMP_TIME" <<'PY'
import sys
src, tmp, decks, f1, f2 = sys.argv[1:6]
t = open(src).read()
rel = 'MUELU_XML_FILE: "xml/multigrid/fluid_solid_ale.xml"'
assert rel in t
t = t.replace(rel, 'MUELU_XML_FILE: "%s/xml/multigrid/fluid_solid_ale.xml"' % decks)
slow1 = 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "y*(1-cos(2*t*pi/5))"'
slow2 = 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "(1-cos(2*t*pi/5))"'
assert slow1 in t and slow2 in t
open(tmp + "/ramp.yaml", "w").write(t)
open(tmp + "/step.yaml", "w").write(
    t.replace(slow1, 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "%s"' % f1, 1)
     .replace(slow2, 'SYMBOLIC_FUNCTION_OF_SPACE_TIME: "%s"' % f2, 1))
PY
echo "STEP_ARM_KEEPS_THE_COSINE_RAMP=$(grep -c '1-cos(2\*t\*pi/5)' "$TMP/step.yaml")"

probe RAMP "$TMP/ramp.yaml"
probe STEP "$TMP/step.yaml"

# The recommended slow ramp: converges, matches the pinned results.
grep -m1 -F "processor 0 finished normally" "$TMP/RAMP.log"
grep -m1 -F "OK (9)" "$TMP/RAMP.log"

# The step: same number of time steps, no divergence anywhere.
echo "RAMP_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/RAMP.log")"
echo "STEP_TIME_STEPS_RUN=$(grep -c '^TIME:' "$TMP/STEP.log")"
echo "RAMP_NONCONVERGENCE_MESSAGES=$(grep -ciE 'did not converge|diverge|unconverged' "$TMP/RAMP.log")"
echo "STEP_NONCONVERGENCE_MESSAGES=$(grep -ciE 'did not converge|diverge|unconverged' "$TMP/STEP.log")"

R=$(grep -c 'Nonlinear Solver Step' "$TMP/RAMP.log")
S=$(grep -c 'Nonlinear Solver Step' "$TMP/STEP.log")
echo "NEWTON_STEPS_SLOW_RAMP=$R"
echo "NEWTON_STEPS_INSTANT_STEP=$S"
if [ "$S" -gt "$R" ]; then
  echo "VERDICT: STEP_INFLOW_COSTS_MORE_NEWTON_STEPS=yes"
else
  echo "VERDICT: STEP_INFLOW_COSTS_MORE_NEWTON_STEPS=no"
fi

# What actually ends the step run is a changed answer, not a failed solve.
grep -m1 -F "Result check failed with 9 errors out of 9 tests" "$TMP/STEP.log"
grep -m1 -E "dispx +at node +38.*is WRONG --> actresult=" "$TMP/STEP.log"
exit 0
