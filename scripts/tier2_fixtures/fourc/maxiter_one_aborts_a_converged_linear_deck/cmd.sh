#!/bin/bash
# Tier-2 for fourc::structural_mechanics#5 — MAXITER: 1 is an abort, not an
# early exit, and it kills a linear deck that has already converged.
#
# The decisive evidence is inside the final status block of the MAXITER: 1 run:
#
#   Converged....Structure-F-Norm = 4.030e-16 < 1.000e-11
#   Failed.......Number of Iterations = 1 < 1
#
# The residual test PASSES and the run still fails, because the iteration
# counter reaches the cap in the same sweep.  MAXITER: 2 on the identical deck
# exits 0.  So there is no such thing as "one iteration is enough for a linear
# problem" — the cheap setting is a generous cap, since Newton stops at the
# tolerance and not at the cap: MAXITER 30 spends the same two iterations.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = KINEM, $2 = MAXITER, $3 = out file
cat > "$3" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-12
  TOLRES: 1.0e-11
  MAXITER: $2
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
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
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0.0, 10.0, 0.0]
    FUNCT: [0, 1, 0]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 1"
  - "NODE 8 DSURFACE 1"
  - "NODE 2 DSURFACE 2"
  - "NODE 3 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM $1"
YAML
}

deck linear    1  "$TMP/lin_1.4C.yaml"
deck linear    2  "$TMP/lin_2.4C.yaml"
deck linear    30 "$TMP/lin_30.4C.yaml"
deck nonlinear 1  "$TMP/nln_1.4C.yaml"
deck nonlinear 30 "$TMP/nln_30.4C.yaml"

probe LIN_MAXITER1  "$TMP/lin_1.4C.yaml"
probe LIN_MAXITER2  "$TMP/lin_2.4C.yaml"
probe LIN_MAXITER30 "$TMP/lin_30.4C.yaml"
probe NLN_MAXITER1  "$TMP/nln_1.4C.yaml"
probe NLN_MAXITER30 "$TMP/nln_30.4C.yaml"

# The cap is what fails, and it fails both kinematics settings.
grep -m1 -F "Failed.......Number of Iterations = 1 < 1" "$TMP/LIN_MAXITER1.log"
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/LIN_MAXITER1.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/LIN_MAXITER1.log"
grep -m1 -F "Failed.......Number of Iterations = 1 < 1" "$TMP/NLN_MAXITER1.log"

# ...while the residual test in that very same status block has PASSED.
python3 - "$TMP/LIN_MAXITER1.log" <<'PY'
import re, sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
block = t.split("-- Final Status Test Results --")[-1]
m = re.search(r"Converged\.+Structure-F-Norm = ([0-9.e+-]+) < ([0-9.e+-]+)", block)
print("MAXITER1_RESIDUAL_TEST_PASSED=%s" % ("yes" if m else "no"))
if m:
    print("MAXITER1_RESIDUAL_BELOW_TOL=%s"
          % ("yes" if float(m.group(1)) < float(m.group(2)) else "no"))
print("MAXITER1_ITERATION_TEST_FAILED=%s"
      % ("yes" if "Failed.......Number of Iterations" in block else "no"))
PY

# Raising the cap by one is enough, and a generous cap costs nothing.
grep -m1 -F "processor 0 finished normally" "$TMP/LIN_MAXITER2.log"
python3 - "$TMP/LIN_MAXITER2.log" "$TMP/LIN_MAXITER30.log" "$TMP/NLN_MAXITER30.log" <<'PY'
import re, sys
def iters(p):
    t = open(p, "rb").read().decode("utf-8", "replace")
    return sum(int(m) for m in re.findall(r"nlniter (\d+)", t))
a, b, c = (iters(p) for p in sys.argv[1:4])
print("ITERS_LIN_MAXITER2=%d" % a)
print("ITERS_LIN_MAXITER30=%d" % b)
print("ITERS_NLN_MAXITER30=%d" % c)
print("GENEROUS_CAP_COSTS_EXTRA_ITERATIONS=%s" % ("yes" if b > a else "no"))
PY
exit 0
