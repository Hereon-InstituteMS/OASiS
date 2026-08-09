#!/bin/bash
# Tier-2 for fourc::contact#8 — when a contact Newton fails, the thing that
# fixes it is a smaller load step, not a smaller PENALTYPARAM.
#
# One two-block mortar penalty deck (two unit cubes 0.1 apart, upper one pushed
# down), nine runs:
#
#   -0.9, TIMESTEP 0.1,  MAXITER 50  -> MAXITER exhausted after ONE step
#   -0.9, TIMESTEP 0.01, MAXITER 50  -> all 100 steps complete, exit 0
#   -0.9, TIMESTEP 0.1,  MAXITER 200 -> still exhausted, now at 200
#   -0.9, TIMESTEP 0.1,  PENALTY 1e2 -> converges, but with more penetration
#   -0.5 / -0.7 / -1.1 at TIMESTEP 0.1 -> fail / converge / converge
#
# The failure mode is active-set chatter, which is why MAXITER does not help:
# the trace alternates forever between an active set of four and an empty one,
# the residual alternates with it, and the update norm dx is pinned at a
# constant.  It is also why the failure is not monotone in the load — a LARGER
# prescribed displacement converges where a smaller one does not.
#
# The last pair prices the alternative remedy: both the small-step run and the
# soft-penalty run reach the full load, and the soft-penalty one gets there by
# letting the bodies interpenetrate an order of magnitude further.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 NUMSTEP, $2 TIMESTEP, $3 dispz, $4 PENALTYPARAM, $5 MAXITER,
          # $6 = "probe" to append gap-measuring result tests, $7 out
RT=""
if [ "$6" = "probe" ]; then
RT='RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 5
      QUANTITY: "dispz"
      VALUE: -1.0e+30
      TOLERANCE: 1.0e-14
  - STRUCTURE:
      DIS: "structure"
      NODE: 9
      QUANTITY: "dispz"
      VALUE: -1.0e+30
      TOLERANCE: 1.0e-14
'
fi
cat > "$7" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: $2
  NUMSTEP: $1
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: $5
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: $4
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
printf '%s' "$RT" >> "$7"
}

deck 10  0.1  "-0.9" 1.0e4 50  plain "$TMP/fail_dt01.yaml"
deck 100 0.01 "-0.9" 1.0e4 50  plain "$TMP/ok_dt001.yaml"
deck 10  0.1  "-0.9" 1.0e4 200 plain "$TMP/maxiter200.yaml"
deck 10  0.1  "-0.9" 1.0e2 50  plain "$TMP/penalty1e2.yaml"
deck 10  0.1  "-0.5" 1.0e4 50  plain "$TMP/load05.yaml"
deck 10  0.1  "-0.7" 1.0e4 50  plain "$TMP/load07.yaml"
deck 10  0.1  "-1.1" 1.0e4 50  plain "$TMP/load11.yaml"
deck 100 0.01 "-0.9" 1.0e4 50  probe "$TMP/gap_stiff.yaml"
deck 10  0.1  "-0.9" 1.0e2 50  probe "$TMP/gap_soft.yaml"

probe FAIL_DT01     "$TMP/fail_dt01.yaml"
probe OK_DT001      "$TMP/ok_dt001.yaml"
probe MAXITER200    "$TMP/maxiter200.yaml"
probe PENALTY1E2    "$TMP/penalty1e2.yaml"
probe LOAD_MINUS_05 "$TMP/load05.yaml"
probe LOAD_MINUS_07 "$TMP/load07.yaml"
probe LOAD_MINUS_11 "$TMP/load11.yaml"
probe GAP_STIFF     "$TMP/gap_stiff.yaml"
probe GAP_SOFT      "$TMP/gap_soft.yaml"

for a in FAIL_DT01 OK_DT001 MAXITER200 PENALTY1E2 \
         LOAD_MINUS_05 LOAD_MINUS_07 LOAD_MINUS_11 GAP_STIFF GAP_SOFT; do
  echo "STEPS_$a=$(grep -c 'Finalised step' "$TMP/$a.log")"
done

# Shrinking the time step is what fixes it.
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/FAIL_DT01.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/FAIL_DT01.log"
grep -m1 -F "Failed.......Number of Iterations = 50 < 50" "$TMP/FAIL_DT01.log"
grep -m1 -F "processor 0 finished normally" "$TMP/OK_DT001.log"
# Raising the cap only moves the number in the message.
grep -m1 -F "Failed.......Number of Iterations = 200 < 200" "$TMP/MAXITER200.log"

# The failure is active-set chatter, not a stiff-system stall.
python3 - "$TMP/FAIL_DT01.log" <<'PY'
import re, sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
sets = [int(m) for m in re.findall(r"Contact-Normal-Active-Set-Size = (\d+)", t)]
tail = sets[-20:]
print("ACTIVE_SET_VALUES_SEEN=%s" % sorted(set(tail)))
print("ACTIVE_SET_ALTERNATES=%s"
      % ("yes" if len(tail) >= 8 and all(a != b for a, b in zip(tail, tail[1:]))
         else "no"))
print("ACTIVE_SET_CHANGE_ANNOUNCEMENTS=%d" % t.count("ACTIVE CONTACT SET HAS CHANGED"))
steps = re.findall(r"\|\|F\|\| = ([0-9.e+-]+)\s+step = ([0-9.e+-]+)\s+dx = ([0-9.e+-]+)", t)
dx = [d for _, _, d in steps][-20:]
res = [float(f) for f, _, _ in steps][-20:]
print("UPDATE_NORM_IS_PINNED=%s" % ("yes" if len(set(dx)) == 1 else "no"))
print("RESIDUAL_ALTERNATES_BETWEEN_TWO_VALUES=%s"
      % ("yes" if len(set(res)) == 2 else "no"))
PY

# It is not monotone in the load: -0.7 and -1.1 converge, -0.5 and -0.9 do not.
python3 - "$TMP/LOAD_MINUS_05.log" "$TMP/LOAD_MINUS_07.log" \
          "$TMP/FAIL_DT01.log" "$TMP/LOAD_MINUS_11.log" <<'PY'
import sys
ok = [open(p, "rb").read().decode("utf-8", "replace").count("Finalised step") == 10
      for p in sys.argv[1:5]]
print("CONVERGENCE_BY_LOAD_05_07_09_11=%s" % ["yes" if o else "no" for o in ok])
print("FAILURE_IS_MONOTONE_IN_THE_LOAD=%s"
      % ("no" if (not ok[0] and ok[1] and not ok[2] and ok[3]) else "yes"))
PY

# Lowering the penalty converges too — by letting the blocks interpenetrate.
python3 - "$TMP/GAP_STIFF.log" "$TMP/GAP_SOFT.log" <<'PY'
import re, sys
pat = re.compile(r"dispz\s+at node\s+(\d+)\s+is WRONG --> actresult=\s*([-0-9.e+]+)")
def gap(p):
    t = open(p, "rb").read().decode("utf-8", "replace")
    d = {int(n): float(v) for n, v in pat.findall(t)}
    return 0.1 + d[9] - d[5]
gs, gf = gap(sys.argv[1]), gap(sys.argv[2])
print("GAP_SMALL_TIMESTEP_STIFF_PENALTY=%.6f" % gs)
print("GAP_BIG_TIMESTEP_SOFT_PENALTY=%.6f" % gf)
print("BOTH_ARE_PENETRATING=%s" % ("yes" if gs < 0 and gf < 0 else "no"))
print("SOFT_PENALTY_PENETRATES_FURTHER=%s" % ("yes" if gf < gs else "no"))
print("SOFT_PENALTY_PENETRATION_RATIO_OVER_10=%s"
      % ("yes" if abs(gf) > 10 * abs(gs) else "no"))
PY
exit 0
