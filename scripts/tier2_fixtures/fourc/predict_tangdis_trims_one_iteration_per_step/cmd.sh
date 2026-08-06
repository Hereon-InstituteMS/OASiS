#!/bin/bash
# Tier-2 for fourc::structural_mechanics#6 — PREDICT: TangDis buys iterations,
# not convergence.
#
# Four-element geometrically nonlinear cantilever (SOLID HEX8, KINEM nonlinear),
# tip Neumann ramped over ten steps, two load levels, TangDis against ConstDis.
# The result test is pinned to the SAME value for both predictors and both pass,
# so the predictor changes only the path to equilibrium, never the equilibrium.
#
# What it costs: exactly one extra Newton iteration per step for ConstDis at
# every step of both load levels.  That is the "step or two" the corrected entry
# describes — a 20% surcharge, not a change of order, and nothing that would
# rescue a diverging solve.
. "$(dirname "$0")/../_lib/preamble.sh"

nodes() {
python3 - <<'PY'
for i in range(5):
    b = 4 * i
    print('  - "NODE %d COORD %.1f 0.0 0.0"' % (b + 1, i))
    print('  - "NODE %d COORD %.1f 1.0 0.0"' % (b + 2, i))
    print('  - "NODE %d COORD %.1f 1.0 1.0"' % (b + 3, i))
    print('  - "NODE %d COORD %.1f 0.0 1.0"' % (b + 4, i))
PY
}
eles() {
python3 - <<'PY'
for e in range(4):
    b = 4 * e
    print('  - "%d SOLID HEX8 %s MAT 1 KINEM nonlinear"'
          % (e + 1, " ".join(str(b + k) for k in range(1, 9))))
PY
}

deck() {  # $1 PREDICT, $2 load, $3 expected tip dispy, $4 outfile
cat > "$4" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  PREDICT: "$1"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 50
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
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 17 DSURFACE 2"
  - "NODE 18 DSURFACE 2"
  - "NODE 19 DSURFACE 2"
  - "NODE 20 DSURFACE 2"
NODE COORDS:
$(nodes)
STRUCTURE ELEMENTS:
$(eles)
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 17
      QUANTITY: "dispy"
      VALUE: $3
      TOLERANCE: 1.0e-10
YAML
}

deck TangDis  100.0 4.39899542151476375e+00 "$TMP/tang_100.4C.yaml"
deck ConstDis 100.0 4.39899542151476375e+00 "$TMP/const_100.4C.yaml"
deck TangDis  600.0 5.80711867329991538e+00 "$TMP/tang_600.4C.yaml"
deck ConstDis 600.0 5.80711867329991538e+00 "$TMP/const_600.4C.yaml"

probe TANG_100  "$TMP/tang_100.4C.yaml"
probe CONST_100 "$TMP/const_100.4C.yaml"
probe TANG_600  "$TMP/tang_600.4C.yaml"
probe CONST_600 "$TMP/const_600.4C.yaml"

# All four reach the same equilibrium — the pinned result test passes each time.
grep -m1 -F "processor 0 finished normally" "$TMP/TANG_100.log"
grep -m1 -F "processor 0 finished normally" "$TMP/CONST_100.log"
for a in TANG_100 CONST_100 TANG_600 CONST_600; do
  echo "RESULT_FAILURES_$a=$(grep -c 'is WRONG --> actresult=' "$TMP/$a.log")"
done

python3 - "$TMP/TANG_100.log" "$TMP/CONST_100.log" \
          "$TMP/TANG_600.log" "$TMP/CONST_600.log" <<'PY'
import re, sys
def its(p):
    t = open(p, "rb").read().decode("utf-8", "replace")
    return [int(m) for m in re.findall(r"nlniter (\d+)", t)]
t1, c1, t6, c6 = (its(p) for p in sys.argv[1:5])
for name, v in (("TANG_100", t1), ("CONST_100", c1),
                ("TANG_600", t6), ("CONST_600", c6)):
    print("STEPS_%s=%d" % (name, len(v)))
    print("TOTAL_ITERS_%s=%d" % (name, sum(v)))
d1 = [b - a for a, b in zip(t1, c1)]
d6 = [b - a for a, b in zip(t6, c6)]
print("PER_STEP_PENALTY_100=%s" % sorted(set(d1)))
print("PER_STEP_PENALTY_600=%s" % sorted(set(d6)))
print("CONSTDIS_COSTS_EXACTLY_ONE_MORE_PER_STEP=%s"
      % ("yes" if set(d1) == {1} and set(d6) == {1} else "no"))
worst = max(sum(c1) / sum(t1), sum(c6) / sum(t6))
print("CONSTDIS_ITERATION_RATIO_AT_LEAST_2X=%s"
      % ("yes" if worst >= 2.0 else "no"))
PY
exit 0
