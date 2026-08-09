#!/bin/bash
# Tier-2 for fourc::contact#9 — a line search does NOT rescue a failing contact
# Newton, and the reason is visible in the trace: the step length never leaves
# 1.0, so there is nothing for the line search to do.
#
# The same failing two-block mortar penalty deck as contact#8 (-0.9 over ten
# steps, PENALTYPARAM 1e4), with STRUCT NOX/Line Search Method set four ways:
#
#   Backtrack / Polynomial / Full Step / More'-Thuente
#       -> byte-identical per-iteration traces, 55 lines reading
#          "step = 1.00000e+00", one finalised step, then
#          "The nonlinear solver did not converge!"
#
# The fifth arm rules out the obvious alternative explanation — that the section
# is being ignored: a bogus Method value is rejected with 'Could not match this
# input', so 4C is reading it and honouring it.  It simply has no leverage on an
# active set that chatters.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = STRUCT NOX/Line Search Method, $2 = out
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
STRUCT NOX/Line Search:
  Method: "$1"
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
    VAL: [0.0, 0.0, -0.9]
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

deck "Backtrack"      "$TMP/backtrack.yaml"
deck "Polynomial"     "$TMP/polynomial.yaml"
deck "Full Step"      "$TMP/fullstep.yaml"
deck "More'-Thuente"  "$TMP/morethuente.yaml"
deck "Sledgehammer"   "$TMP/bogus.yaml"

probe LS_BACKTRACK   "$TMP/backtrack.yaml"
probe LS_POLYNOMIAL  "$TMP/polynomial.yaml"
probe LS_FULLSTEP    "$TMP/fullstep.yaml"
probe LS_MORETHUENTE "$TMP/morethuente.yaml"
probe LS_BOGUS       "$TMP/bogus.yaml"

for a in LS_BACKTRACK LS_POLYNOMIAL LS_FULLSTEP LS_MORETHUENTE; do
  echo "STEPS_$a=$(grep -c 'Finalised step' "$TMP/$a.log")"
  echo "UNIT_STEP_LINES_$a=$(grep -c 'step = 1.00000e+00' "$TMP/$a.log")"
done
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/LS_BACKTRACK.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/LS_MORETHUENTE.log"

# The STRUCT NOX section IS read: a bogus Method is refused.
grep -m1 -F "Could not match this input" "$TMP/LS_BOGUS.log"
echo "BOGUS_METHOD_STARTED_A_SOLVE=$(grep -c 'Finalised step' "$TMP/LS_BOGUS.log")"

python3 - "$TMP/LS_BACKTRACK.log" "$TMP/LS_POLYNOMIAL.log" \
          "$TMP/LS_FULLSTEP.log" "$TMP/LS_MORETHUENTE.log" <<'PY'
import re, sys
pat = re.compile(r"\|\|F\|\| = ([0-9.e+-]+)\s+step = ([0-9.e+-]+)\s+dx = ([0-9.e+-]+)")
traces = []
for p in sys.argv[1:5]:
    t = open(p, "rb").read().decode("utf-8", "replace")
    traces.append(pat.findall(t))
print("TRACES_IDENTICAL_ACROSS_METHODS=%s"
      % ("yes" if all(x == traces[0] for x in traces) and traces[0] else "no"))
steps = sorted({s for _, s, _ in traces[0]})
print("STEP_LENGTHS_SEEN=%s" % steps)
print("STEP_LENGTHS_STRICTLY_BETWEEN_ZERO_AND_ONE=%d"
      % sum(1 for s in steps if 0.0 < float(s) < 1.0))
PY
exit 0
