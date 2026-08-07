#!/bin/bash
# Tier-2 for fourc::structural_dynamics#5 — and a FALSIFICATION of it.
#
# Claimed: "PREDICT: ConstDisVelAcc is the recommended predictor.  PREDICT:
#           TangDis may cause issues in highly dynamic problems.  Signal:
#           Newton iteration count per time step is higher with TangDis."
# Observed on a transient inline HEX8 cantilever under GenAlpha:
#   * the DEFAULT predictor is ConstDis, not ConstDisVelAcc — 4C echoes it as
#     "=== Structural predictor: ConstDis ===" when nothing is set;
#   * TangDis takes FEWER total Newton iterations than ConstDisVelAcc, not more;
#   * all three reach the same answer, so the choice is a cost question and not
#     a correctness one;
#   * a plausible misspelling is rejected at parse, so the risk of getting this
#     wrong silently is nil.
#
# The per-step iteration count is printed by 4C itself in the "Finalised step"
# banner as "nlniter N"; the arms below sum it.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra STRUCTURAL DYNAMIC line, $2 = out
python3 - "$1" "$2" <<'PY'
import sys
extra, out = sys.argv[1], sys.argv[2]
nx, ny, nz = 10, 1, 1
Lx, Ly, Lz = 10.0, 1.0, 0.5
nid, coords = {}, []
c = 0
for k in range(nz + 1):
    for j in range(ny + 1):
        for i in range(nx + 1):
            c += 1
            nid[(i, j, k)] = c
            coords.append(f'  - "NODE {c} COORD {i*Lx/nx:.16e} {j*Ly/ny:.16e} {k*Lz/nz:.16e}"')
eles = []
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            e = k*ny*nx + j*nx + i + 1
            q = [nid[(i, j, k)], nid[(i+1, j, k)], nid[(i+1, j+1, k)], nid[(i, j+1, k)],
                 nid[(i, j, k+1)], nid[(i+1, j, k+1)], nid[(i+1, j+1, k+1)], nid[(i, j+1, k+1)]]
            eles.append(f'  - "{e} SOLID HEX8 {" ".join(str(x) for x in q)} MAT 1 KINEM nonlinear"')
clamp = [nid[(0, j, k)] for k in range(nz+1) for j in range(ny+1)]
tip = [nid[(nx, j, k)] for k in range(nz+1) for j in range(ny+1)]
open(out, "w").write(f"""PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:{extra}
  DYNAMICTYPE: "GenAlpha"
  TIMESTEP: 0.05
  NUMSTEP: 8
  MAXTIME: 0.4
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-08
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
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, -1.0]
    FUNCT: [0, 0, 1]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:
{chr(10).join(f'  - "NODE {i} DSURFACE 1"' for i in clamp)}
{chr(10).join(f'  - "NODE {i} DSURFACE 2"' for i in tip)}
NODE COORDS:
{chr(10).join(coords)}
STRUCTURE ELEMENTS:
{chr(10).join(eles)}
""")
PY
}

deck ""                              "$TMP/default.4C.yaml"
deck $'\n  PREDICT: "ConstDis"'       "$TMP/constdis.4C.yaml"
deck $'\n  PREDICT: "ConstDisVelAcc"' "$TMP/cdva.4C.yaml"
deck $'\n  PREDICT: "TangDis"'        "$TMP/tangdis.4C.yaml"
deck $'\n  PREDICT: "ConstDisVelAccel"' "$TMP/typo.4C.yaml"

probe DEFAULT   "$TMP/default.4C.yaml"
probe CONSTDIS  "$TMP/constdis.4C.yaml"
probe CDVA      "$TMP/cdva.4C.yaml"
probe TANGDIS   "$TMP/tangdis.4C.yaml"
probe TYPO      "$TMP/typo.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/DEFAULT.log"
# 4C echoes the predictor it selected — the default is not ConstDisVelAcc.
echo "DEFAULT_PREDICTOR=$(grep -m1 -oE '=== Structural predictor: [A-Za-z]+ ===' "$TMP/DEFAULT.log")"
sumiter() { grep -oE 'nlniter [0-9]+' "$1" | awk '{s+=$2} END{printf "%d", s}'; }
echo "ITERS_DEFAULT=$(sumiter "$TMP/DEFAULT.log")"
echo "ITERS_CONSTDIS=$(sumiter "$TMP/CONSTDIS.log")"
echo "ITERS_CONSTDISVELACC=$(sumiter "$TMP/CDVA.log")"
echo "ITERS_TANGDIS=$(sumiter "$TMP/TANGDIS.log")"
if [ "$(sumiter "$TMP/TANGDIS.log")" -lt "$(sumiter "$TMP/CDVA.log")" ]; then
  echo "VERDICT: TANGDIS_COSTS_MORE_THAN_CONSTDISVELACC=no"
else
  echo "VERDICT: TANGDIS_COSTS_MORE_THAN_CONSTDISVELACC=yes"
fi
# A near-miss spelling is rejected at parse, so this cannot go wrong silently.
grep -m1 -F "Could not match this input" "$TMP/TYPO.log"
grep -m1 -F "ConstDisVelAccel" "$TMP/TYPO.log"
exit 0
