#!/bin/bash
# Tier-2 for fourc::brownian_dynamics#0 — how you may and may not pick the
# Brownian time step.
#
# The entry's Signal (mean-square displacement scaling wrongly) is an ensemble
# statement and cannot be read off one 4C run.  What CAN be shown in seconds is
# the operational consequence, and it is the thing that actually catches people:
# a thermal run does not step-converge, so the usual "halve dt and see if the
# answer stops moving" test is meaningless here and will make you accept any dt.
#
# Four arms on upstream's periodic-RVE filament deck, physical end time fixed at
# 1 and RANDSEED fixed, so every run is deterministic and reproducible:
#
#   KT = 0    dt = 1e-2 (100 steps) and dt = 5e-3 (200 steps)
#             -> node 2's TRANSVERSE displacement stays at machine zero in both,
#                moving by 7e-17 under refinement, as a deterministic run should.
#                (The refined cold arm still trips one result test, on the AXIAL
#                component: that one is driven by the prescribed ramp and is
#                sampled at different instants, which is ordinary discretisation
#                error and nothing to do with noise.  The thermal comparison
#                below therefore uses dispx only.)
#   KT > 0    dt = 1e-2, 5e-3, 2.5e-3
#             -> 1.97729908169594938e-02, 1.41182827349665852e-02,
#                -9.93856384578826677e-04: the value moves by more than its own
#                size under refinement and changes SIGN
#
# Refining dt does not converge the trajectory, it draws a different realisation.
# dt has to be judged against the physical relaxation time, never against a
# step-refinement study.
[ -d /media/alexander/PortableSSD/audit4c/tier2_scratch ] && \
  export TMPDIR=/media/alexander/PortableSSD/audit4c/tier2_scratch
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream beam3r_line2_backweuler_browndyn_periodic_rve_dirich_element.4C.yaml) || exit 3
grep -q "  TIMESTEP: 0.01" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "  NUMSTEP: 100"   "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "RANDSEED: 1"      "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

arm() {  # $1 = KT, $2 = dt, $3 = numstep, $4 = label
  python3 - "$BASE" "$TMP/$4.yaml" "$1" "$2" "$3" <<'PY'
import sys
t = open(sys.argv[1]).read()
blk = "BROWNIAN DYNAMICS:\n  BROWNDYNPROB: true\n  VISCOSITY: 0.001\n"
assert blk in t
if float(sys.argv[3]) > 0.0:
    t = t.replace(blk, blk + "  KT: %s\n" % sys.argv[3])
t = t.replace("  TIMESTEP: 0.01", "  TIMESTEP: %s" % sys.argv[4])
t = t.replace("  NUMSTEP: 100", "  NUMSTEP: %s" % sys.argv[5])
open(sys.argv[2], "w").write(t)
PY
  probe "$4" "$TMP/$4.yaml"
}

arm 0.0      0.01   100 COLD_DT1
arm 0.0      0.005  200 COLD_DT2
arm 4.14e-06 0.01   100 HOT_DT1
arm 4.14e-06 0.005  200 HOT_DT2
arm 4.14e-06 0.0025 400 HOT_DT3

grep -m1 -F "processor 0 finished normally" "$TMP/COLD_DT1.log"
echo "COLD_DT1_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/COLD_DT1.log")"
echo "COLD_DT2_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/COLD_DT2.log")"

# The thermal arms all leave the deterministic reference, each differently.
grep -m1 -F "is WRONG --> actresult= 1.97729908169594938e-02" "$TMP/HOT_DT1.log"
grep -m1 -F "is WRONG --> actresult= 1.41182827349665852e-02" "$TMP/HOT_DT2.log"
grep -m1 -F "is WRONG --> actresult=-9.93856384578826677e-04" "$TMP/HOT_DT3.log"

python3 - "$TMP" <<'PY'
import re, sys, pathlib
tmp = pathlib.Path(sys.argv[1])
def dispx(label):
    for l in (tmp / (label + ".log")).read_text().splitlines():
        m = re.search(r"dispx\s+at node\s+2\s+is (?:WRONG --> actresult=\s*(-?[0-9.e+-]+)|CORRECT, abs\(diff\)=\s*([0-9.e+-]+))", l)
        if m:
            return float(m.group(1) if m.group(1) else m.group(2))
    return None
c1, c2 = dispx("COLD_DT1"), dispx("COLD_DT2")
h1, h2, h3 = dispx("HOT_DT1"), dispx("HOT_DT2"), dispx("HOT_DT3")
print("COLD_REFINEMENT_CHANGE=%.3e" % abs(c1 - c2))
print("HOT_REFINEMENT_CHANGE_1_TO_2=%.3e" % abs(h1 - h2))
print("HOT_REFINEMENT_CHANGE_2_TO_3=%.3e" % abs(h2 - h3))
print("DETERMINISTIC_RUN_STEP_CONVERGES=%s" % ("yes" if abs(c1 - c2) < 1e-12 else "no"))
print("THERMAL_RUN_STEP_CONVERGES=%s"
      % ("no" if abs(h2 - h3) > 0.5 * abs(h2) else "yes"))
print("THERMAL_REFINEMENT_FLIPS_SIGN=%s" % ("yes" if h2 * h3 < 0 else "no"))
PY
exit 0
