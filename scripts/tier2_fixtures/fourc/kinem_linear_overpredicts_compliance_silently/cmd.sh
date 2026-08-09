#!/bin/bash
# Tier-2 for fourc::structural_mechanics#14 — KINEM linear on an ordinary
# St-Venant-Kirchhoff solid never errors.  It drops the geometric nonlinearity
# and the run looks perfectly healthy, so the only defence is a comparison.
#
# One HEX8 unit cube, YOUNG 1000 NUE 0.3, surface Neumann in y, two load levels
# a factor 300 apart.  Each linear arm carries its NONLINEAR twin's result test,
# pinned at 1e-12, at TWO probe nodes at once — node 6 and node 3 — because the
# size of the discrepancy is node-dependent and a single number without a node
# is meaningless.
#
#   small load: the two agree to 0.06%, and the SIGN differs between the two
#               probe nodes — linear is the smaller value at node 6 and the
#               larger one at node 3, so the sign tells you nothing at small
#               strain
#   300x load : linear over-predicts by 5.7% at node 6 and by 81.8% at node 3
#
# Both linear runs exit through the added result test and nothing else; a deck
# without a result test finishes with no warning at all.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 KINEM, $2 load, $3 ref node6, $4 ref node3, $5 out
cat > "$5" <<YAML
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
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: $4
      TOLERANCE: 1.0e-12
YAML
}

SMALL6=4.48471241704061219e-03
SMALL3=4.47909266337460053e-03
BIG6=1.27200961795990475e+00
BIG3=7.39435436319622608e-01

deck nonlinear 1.0   "$SMALL6" "$SMALL3" "$TMP/nl_small.yaml"
deck linear    1.0   "$SMALL6" "$SMALL3" "$TMP/l_small.yaml"
deck nonlinear 300.0 "$BIG6"   "$BIG3"   "$TMP/nl_big.yaml"
deck linear    300.0 "$BIG6"   "$BIG3"   "$TMP/l_big.yaml"

probe NL_SMALL  "$TMP/nl_small.yaml"
probe LIN_SMALL "$TMP/l_small.yaml"
probe NL_BIG    "$TMP/nl_big.yaml"
probe LIN_BIG   "$TMP/l_big.yaml"

# The two nonlinear references are healthy and self-consistent.
grep -m1 -F "processor 0 finished normally" "$TMP/NL_BIG.log"
echo "RESULT_FAILURES_NL_SMALL=$(grep -c 'is WRONG --> actresult=' "$TMP/NL_SMALL.log")"
echo "RESULT_FAILURES_NL_BIG=$(grep -c 'is WRONG --> actresult=' "$TMP/NL_BIG.log")"

python3 - "$TMP/LIN_SMALL.log" "$TMP/LIN_BIG.log" <<'PY'
import re, sys
pat = re.compile(r"dispy\s+at node\s+(\d+)\s+is WRONG --> actresult=\s*([-0-9.e+]+),"
                 r" givenresult=\s*([-0-9.e+]+)")
for tag, p in zip(("SMALL", "BIG"), sys.argv[1:3]):
    t = open(p, "rb").read().decode("utf-8", "replace")
    hits = {int(n): (float(a), float(g)) for n, a, g in pat.findall(t)}
    for node in (6, 3):
        if node in hits:
            a, g = hits[node]
            print("LINEAR_VS_NONLINEAR_PERCENT_%s_NODE%d=%+.2f"
                  % (tag, node, 100.0 * (a - g) / g))
        else:
            print("LINEAR_VS_NONLINEAR_PERCENT_%s_NODE%d=agrees" % (tag, node))
    blocks = [l for l in t.split("\n") if "PROC 0 ERROR" in l]
    print("%s_ONLY_ERROR_IS_THE_RESULT_TEST=%s"
          % (tag, "yes" if blocks and all("4C_utils_result_test.cpp" in l
                                          for l in blocks) else "no"))
    print("%s_MENTIONS_KINEM=%d" % (tag, t.count("KINEM")))
PY
exit 0
