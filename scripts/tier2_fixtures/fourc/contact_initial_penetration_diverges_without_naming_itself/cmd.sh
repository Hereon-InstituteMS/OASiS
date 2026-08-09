#!/bin/bash
# Tier-2 for fourc::contact#7 — contact surfaces that already overlap in the
# reference configuration break the first Newton solve, and 4C never says why.
#
# Three arms of the same two-block mortar penalty deck, differing only in where
# the upper block starts:
#
#   z = 1.1  gap 0.1   -> converges, ten steps, exits 0
#   z = 1.0  touching  -> converges, ten steps, exits 0
#   z = 0.9  overlap 0.1 -> "The nonlinear solver did not converge!" before a
#                           single step is finalised
#
# The half of the entry that IS observable is the residual: at Nonlinear Solver
# Step 0 the overlapping deck reports a force norm an order of magnitude above
# the touching one, because the penalty is already pushing on a closed gap. The
# fixture asserts that ordering rather than a fixed number.
#
# The half that is not: 'initial penetration X.X > tolerance'. No such
# MortarInterface diagnostic exists — the log names neither penetration nor
# overlap nor the interface, so nothing distinguishes this from any other
# non-convergence. INITIAL_PENETRATION_IS_NAMED=no is the point of the fixture.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = z of the upper block's bottom face, $2 = z of its top face
cat <<YAML
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
    VAL: [0.0, 0.0, -0.3]
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
  - "NODE 9 COORD 0.0 0.0 $1"
  - "NODE 10 COORD 1.0 0.0 $1"
  - "NODE 11 COORD 1.0 1.0 $1"
  - "NODE 12 COORD 0.0 1.0 $1"
  - "NODE 13 COORD 0.0 0.0 $2"
  - "NODE 14 COORD 1.0 0.0 $2"
  - "NODE 15 COORD 1.0 1.0 $2"
  - "NODE 16 COORD 0.0 1.0 $2"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
  - "2 SOLID HEX8 9 10 11 12 13 14 15 16 MAT 1 KINEM nonlinear"
YAML
}

deck 1.1 2.1 > "$TMP/gap.yaml"
deck 1.0 2.0 > "$TMP/touching.yaml"
deck 0.9 1.9 > "$TMP/overlap.yaml"

probe GAP      "$TMP/gap.yaml"
probe TOUCHING "$TMP/touching.yaml"
probe OVERLAP  "$TMP/overlap.yaml"

# A gap and an exactly-touching start are both fine.
grep -m1 -F "processor 0 finished normally" "$TMP/GAP.log"
grep -m1 -F "processor 0 finished normally" "$TMP/TOUCHING.log"
echo "GAP_STEPS=$(grep -c 'Finalised step' "$TMP/GAP.log")"
echo "TOUCHING_STEPS=$(grep -c 'Finalised step' "$TMP/TOUCHING.log")"

# An overlap of one tenth of an element is not.
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/OVERLAP.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/OVERLAP.log"
echo "OVERLAP_STEPS=$(grep -c 'Finalised step' "$TMP/OVERLAP.log")"
# It starts with a full active set, before any load has been applied.
grep -m1 -F "Contact-Normal-Active-Set-Size = 4" "$TMP/OVERLAP.log"

python3 - "$TMP/TOUCHING.log" "$TMP/OVERLAP.log" <<'PY'
import re, sys
def first_residual(path):
    t = open(path, "rb").read().decode("utf-8", "replace")
    m = re.search(r'\|\|F\|\| = ([0-9.eE+-]+)', t)
    assert m, "no NOX residual line in " + path
    return float(m.group(1))
touch, over = first_residual(sys.argv[1]), first_residual(sys.argv[2])
print("STEP0_RESIDUAL_TOUCHING=%.6e" % touch)
print("STEP0_RESIDUAL_OVERLAP=%.6e" % over)
print("OVERLAP_STARTS_WITH_LARGER_RESIDUAL=%s" % ("yes" if over > touch else "no"))
print("OVERLAP_RESIDUAL_IS_AN_ORDER_LARGER=%s" % ("yes" if over > 10.0 * touch else "no"))

log = open(sys.argv[2], "rb").read().decode("utf-8", "replace").lower()
named = ("initial penetration" in log) or ("penetrat" in log) or ("overlap" in log)
print("INITIAL_PENETRATION_IS_NAMED=%s" % ("yes" if named else "no"))
print("CLAIMED_TOLERANCE_TEXT=%d" % log.count("> tolerance"))
PY
exit 0
