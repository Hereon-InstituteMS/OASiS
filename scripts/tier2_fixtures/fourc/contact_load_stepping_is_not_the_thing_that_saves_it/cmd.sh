#!/bin/bash
# Tier-2 for fourc::contact#3 — the entry says quasi-static contact "MUST use
# load stepping" and that applying the full load in one step "almost always
# causes Newton divergence". On this problem it is the other way round.
#
# One two-block mortar penalty deck, two load levels, one and ten steps each:
#
#   -0.3 in 1 step  -> converges, 7 Newton iterations
#   -0.3 in 10 steps-> converges, 41 Newton iterations in total
#   -0.9 in 1 step  -> converges
#   -0.9 in 10 steps-> FAILS: "The nonlinear solver did not converge!"
#
# The harsh case is the decisive one: the single-step run reaches the full load
# and the ten-step run does not. Load stepping is not a safety property of
# contact; the intermediate configurations it creates have their own active-set
# transitions, and one of those is what fails here. The claimed diagnostic —
# "NOX hits StatusTest::MaxIters at the first step" — is not what appears
# either; the run that fails does so after finalising step 1, and MAXITER is
# never the binding test.
#
# What the entry gets right is that a contact solve can fail at a step where the
# contact set changes a lot. What it gets wrong is prescribing more steps as the
# remedy.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = NUMSTEP, $2 = TIMESTEP, $3 = prescribed z displacement
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: $2
  NUMSTEP: $1
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, $3]
    FUNCT: [0, 0, 1]
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
  - "NODE 8 DSURFACE 2"
  - "NODE 9 DSURFACE 3"
  - "NODE 10 DSURFACE 3"
  - "NODE 11 DSURFACE 3"
  - "NODE 12 DSURFACE 3"
  - "NODE 13 DSURFACE 4"
  - "NODE 14 DSURFACE 4"
  - "NODE 15 DSURFACE 4"
  - "NODE 16 DSURFACE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
  - "NODE 9 COORD 0.0 0.0 1.1"
  - "NODE 10 COORD 1.0 0.0 1.1"
  - "NODE 11 COORD 1.0 1.0 1.1"
  - "NODE 12 COORD 0.0 1.0 1.1"
  - "NODE 13 COORD 0.0 0.0 2.1"
  - "NODE 14 COORD 1.0 0.0 2.1"
  - "NODE 15 COORD 1.0 1.0 2.1"
  - "NODE 16 COORD 0.0 1.0 2.1"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
  - "2 SOLID HEX8 9 10 11 12 13 14 15 16 MAT 1 KINEM nonlinear"
YAML
}

deck 1  1.0 "-0.3" > "$TMP/one_mild.yaml"
deck 10 0.1 "-0.3" > "$TMP/ten_mild.yaml"
deck 1  1.0 "-0.9" > "$TMP/one_harsh.yaml"
deck 10 0.1 "-0.9" > "$TMP/ten_harsh.yaml"

probe ONEMILD  "$TMP/one_mild.yaml"
probe TENMILD  "$TMP/ten_mild.yaml"
probe ONEHARSH "$TMP/one_harsh.yaml"
probe TENHARSH "$TMP/ten_harsh.yaml"

# Both single-step runs reach the full load.
grep -m1 -F "processor 0 finished normally" "$TMP/ONEMILD.log"
grep -m1 -F "processor 0 finished normally" "$TMP/ONEHARSH.log"
echo "ONEMILD_STEPS=$(grep -c 'Finalised step' "$TMP/ONEMILD.log")"
echo "ONEHARSH_STEPS=$(grep -c 'Finalised step' "$TMP/ONEHARSH.log")"
echo "TENMILD_STEPS=$(grep -c 'Finalised step' "$TMP/TENMILD.log")"
# The load-stepped harsh run does not.
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/TENHARSH.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/TENHARSH.log"
echo "TENHARSH_STEPS=$(grep -c 'Finalised step' "$TMP/TENHARSH.log")"

python3 - "$TMP/ONEMILD.log" "$TMP/TENMILD.log" "$TMP/ONEHARSH.log" "$TMP/TENHARSH.log" <<'PY'
import re, sys
def iters(p):
    t = open(p, "rb").read().decode("utf-8", "replace")
    return sum(int(m) for m in re.findall(r'nlniter (\d+)', t))
def steps(p):
    t = open(p, "rb").read().decode("utf-8", "replace")
    return t.count("Finalised step")
om, tm, oh, th = sys.argv[1:5]
print("NEWTON_ITERS_ONE_STEP_MILD=%d" % iters(om))
print("NEWTON_ITERS_TEN_STEP_MILD=%d" % iters(tm))
print("ONE_STEP_COSTS_FEWER_ITERATIONS=%s" % ("yes" if iters(om) < iters(tm) else "no"))
print("ONE_STEP_CONVERGES_AT_BOTH_LOADS=%s"
      % ("yes" if steps(om) == 1 and steps(oh) == 1 else "no"))
print("LOAD_STEPPING_RESCUES_THE_HARSH_CASE=%s"
      % ("yes" if steps(th) == 10 else "no"))
PY

# The claimed signal is not what the failing run reports.
python3 - "$TMP/TENHARSH.log" <<'PY'
import sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
print("CLAIMED_MAXITERS_SIGNAL=%d" % t.count("StatusTest::MaxIters"))
print("FAILURE_IS_AT_FIRST_STEP=%s"
      % ("yes" if t.count("Finalised step") == 0 else "no"))
PY
exit 0
