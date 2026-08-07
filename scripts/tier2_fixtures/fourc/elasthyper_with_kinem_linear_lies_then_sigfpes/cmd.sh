#!/bin/bash
# Tier-2 for fourc::structural_mechanics#13 — MAT_ElastHyper with KINEM linear
# is accepted by both the parser and the element factory, and it fails in two
# different ways depending on how far the structure deforms.
#
# One HEX8 unit cube, MAT_ElastHyper = ELAST_CoupNeoHooke + ELAST_VolSussmanBathe,
# surface Neumann in y, tip dispy at node 6.  Each linear arm carries the result
# test of its NONLINEAR twin, pinned at 1e-12, so 4C itself reports the gap:
#
#   load  50 -> linear runs, 1.8% low
#   load 100 -> linear runs, 2.9% low  (the error grows with the load)
#   load 300 -> linear is KILLED by SIGFPE, shell status 136, while the
#               nonlinear twin finishes
#
# Neither failure names KINEM.  The quiet branch has no diagnostic at all beyond
# the result test this fixture adds; the loud branch has no 4C error block
# either, only the signal report, with Mat::ElastHyper::evaluate and
# DisplacementBasedLinearKinematicsFormulation at the top of the stack.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 KINEM, $2 load, $3 reference dispy, $4 out
cat > "$4" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-12
  TOLRES: 1.0e-11
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 2
      MATIDS: [2, 3]
      DENS: 1.0
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 1000.0
      NUE: 0.3
  - MAT: 3
    ELAST_VolSussmanBathe:
      KAPPA: 1000.0
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
    VAL: [0.0, $2, 0.0]
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
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 6
      QUANTITY: "dispy"
      VALUE: $3
      TOLERANCE: 1.0e-12
YAML
}

REF50=2.07913934875450235e-01
REF100=4.20495844658128559e-01
REF300=1.19488815226590517e+00

deck nonlinear 50.0  "$REF50"  "$TMP/nl50.yaml"
deck linear    50.0  "$REF50"  "$TMP/l50.yaml"
deck nonlinear 100.0 "$REF100" "$TMP/nl100.yaml"
deck linear    100.0 "$REF100" "$TMP/l100.yaml"
deck nonlinear 300.0 "$REF300" "$TMP/nl300.yaml"
deck linear    300.0 "$REF300" "$TMP/l300.yaml"

probe NL_50   "$TMP/nl50.yaml"
probe LIN_50  "$TMP/l50.yaml"
probe NL_100  "$TMP/nl100.yaml"
probe LIN_100 "$TMP/l100.yaml"
probe NL_300  "$TMP/nl300.yaml"
probe LIN_300 "$TMP/l300.yaml"

# The three hyperelastic reference runs are healthy at every load.
grep -m1 -F "processor 0 finished normally" "$TMP/NL_300.log"
for a in NL_50 NL_100 NL_300; do
  echo "RESULT_FAILURES_$a=$(grep -c 'is WRONG --> actresult=' "$TMP/$a.log")"
done

# The quiet branch: the linear twin runs and is wrong by a margin that grows.
python3 - "$TMP/LIN_50.log" "$TMP/LIN_100.log" <<'PY'
import re, sys
out = []
for name, p in zip(("50", "100"), sys.argv[1:3]):
    t = open(p, "rb").read().decode("utf-8", "replace")
    m = re.search(r"is WRONG --> actresult=\s*([-0-9.e+]+), givenresult=\s*([-0-9.e+]+)", t)
    act, ref = float(m.group(1)), float(m.group(2))
    pct = 100.0 * (act - ref) / ref
    out.append(pct)
    print("LINEAR_ERROR_PERCENT_AT_LOAD_%s=%.1f" % (name, pct))
    blocks = [l for l in t.split("\n") if "PROC 0 ERROR" in l]
    print("LOAD_%s_ONLY_ERROR_IS_THE_RESULT_TEST=%s"
          % (name, "yes" if blocks and all("4C_utils_result_test.cpp" in l
                                           for l in blocks) else "no"))
print("LINEAR_ERROR_GROWS_WITH_LOAD=%s"
      % ("yes" if abs(out[1]) > abs(out[0]) else "no"))
print("LINEAR_IS_TOO_SOFT_OR_TOO_STIFF=%s"
      % ("too_stiff" if out[0] < 0 else "too_soft"))
PY

# The loud branch: no 4C error block, only a signal, and the stack names the
# material evaluation and the linear kinematics formulation.
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/LIN_300.log"
grep -m1 -F "Signal code: Invalid floating point operation (7)" "$TMP/LIN_300.log"
echo "LIN_300_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/LIN_300.log")"
echo "LIN_300_STACK_NAMES_ELASTHYPER=$(grep -c 'ElastHyper' "$TMP/LIN_300.log")"
echo "LIN_300_STACK_NAMES_LINEAR_KINEMATICS=$(grep -c 'DisplacementBasedLinearKinematicsFormulation' "$TMP/LIN_300.log")"
echo "LIN_300_MENTIONS_KINEM=$(grep -c 'KINEM' "$TMP/LIN_300.log")"
exit 0
