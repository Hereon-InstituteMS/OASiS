#!/bin/bash
# Tier-2 for fourc::structural_mechanics#10 — on the WALL element EAS and
# STRESS_STRAIN move the answer by percent-level amounts, and the two keywords
# fail in opposite ways.
#
# One 1x1 WALL QUAD4 unit square, YOUNG 1000 NUE 0.3 THICK 1.0 GP 2 2, left edge
# clamped, right edge line Neumann VAL 1 in y, dispy read at the loaded corner
# node 2.  Every comparison is at the SAME node and the SAME kinematics.
#
#   KINEM nonlinear, plane_strain, EAS none  -> reference, result test PASSES
#   KINEM nonlinear, plane_strain, EAS full  -> same test FAILS: +23.1%, silently
#   KINEM linear,    plane_strain, EAS none  -> reference, result test PASSES
#   KINEM linear,    plane_stress, EAS none  -> same test FAILS: +6.7%, silently
#   KINEM linear,    plane_strain, EAS full  -> HARD ERROR, exit 1
#   KINEM linear,    plane_stress, EAS full  -> HARD ERROR, exit 1
#
# So the EAS trap is a run that stops and the STRESS_STRAIN trap is a run that
# lies.  The contrast arms are driven by 4C's own result test, which prints the
# actual value it measured, so the percentages below are read out of the log
# rather than asserted from prose.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 KINEM, $2 EAS, $3 STRESS_STRAIN, $4 reference dispy, $5 out
cat > "$5" <<YAML
PROBLEM SIZE:
  DIM: 2
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
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 2
    ONOFF: [0, 1]
    VAL: [0.0, 1.0]
    FUNCT: [0, 1]
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
  - "NODE 2 DLINE 2"
  - "NODE 3 DLINE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM $1 EAS $2 THICK 1.0 STRESS_STRAIN $3 GP 2 2"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 2
      QUANTITY: "dispy"
      VALUE: $4
      TOLERANCE: 1.0e-12
YAML
}

NL_REF=4.33567955849997223e-03
L_REF=4.33333333333346064e-03

deck nonlinear none plane_strain "$NL_REF" "$TMP/nl_none.yaml"
deck nonlinear full plane_strain "$NL_REF" "$TMP/nl_full.yaml"
deck linear    none plane_strain "$L_REF"  "$TMP/l_strain.yaml"
deck linear    none plane_stress "$L_REF"  "$TMP/l_stress.yaml"
deck linear    full plane_strain "$L_REF"  "$TMP/l_full_strain.yaml"
deck linear    full plane_stress "$L_REF"  "$TMP/l_full_stress.yaml"

probe NL_EAS_NONE   "$TMP/nl_none.yaml"
probe NL_EAS_FULL   "$TMP/nl_full.yaml"
probe LIN_STRAIN    "$TMP/l_strain.yaml"
probe LIN_STRESS    "$TMP/l_stress.yaml"
probe LIN_EAS_FULL_STRAIN "$TMP/l_full_strain.yaml"
probe LIN_EAS_FULL_STRESS "$TMP/l_full_stress.yaml"

# The two reference arms run and match to 1e-12.
grep -m1 -F "processor 0 finished normally" "$TMP/NL_EAS_NONE.log"
echo "NL_EAS_NONE_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/NL_EAS_NONE.log")"
echo "LIN_STRAIN_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/LIN_STRAIN.log")"

# The two contrast arms run just as happily and give a different answer, with
# no warning of any kind.
echo "NL_EAS_FULL_WARNS_ABOUT_EAS=$(grep -ci 'eas' "$TMP/NL_EAS_FULL.log")"
echo "LIN_STRESS_WARNS_ABOUT_STRESS_STRAIN=$(grep -ci 'stress_strain\|plane_stress' "$TMP/LIN_STRESS.log")"
# ...and the ONLY thing that stops them is 4C's own result test noticing the
# changed number, which is exactly why a real deck without one says nothing.
python3 - "$TMP/NL_EAS_FULL.log" "$TMP/LIN_STRESS.log" <<'PY'
import sys
names = ("NL_EAS_FULL", "LIN_STRESS")
for name, p in zip(names, sys.argv[1:3]):
    t = open(p, "rb").read().decode("utf-8", "replace")
    blocks = [l for l in t.split("\n") if "PROC 0 ERROR" in l]
    only_result_test = bool(blocks) and all(
        "4C_utils_result_test.cpp" in l for l in blocks)
    print("%s_ONLY_ERROR_IS_THE_RESULT_TEST=%s"
          % (name, "yes" if only_result_test else "no"))
PY

python3 - "$TMP/NL_EAS_FULL.log" "$TMP/LIN_STRESS.log" <<'PY'
import re, sys
def pair(p):
    t = open(p, "rb").read().decode("utf-8", "replace")
    m = re.search(r"is WRONG --> actresult=\s*([-0-9.e+]+), givenresult=\s*([-0-9.e+]+)", t)
    return float(m.group(1)), float(m.group(2))
act, ref = pair(sys.argv[1])
print("EAS_FULL_SHIFT_PERCENT=%.1f" % (100.0 * (act - ref) / ref))
act, ref = pair(sys.argv[2])
print("PLANE_STRESS_SHIFT_PERCENT=%.1f" % (100.0 * (act - ref) / ref))
PY

# EAS with linear kinematics is the loud branch, whatever STRESS_STRAIN says.
grep -m1 -F "ERROR: No EAS for geometrically linear WALL element" "$TMP/LIN_EAS_FULL_STRAIN.log"
grep -m1 -F "ERROR: No EAS for geometrically linear WALL element" "$TMP/LIN_EAS_FULL_STRESS.log"
grep -m1 -F "4C_w1_input.cpp" "$TMP/LIN_EAS_FULL_STRAIN.log"
exit 0
