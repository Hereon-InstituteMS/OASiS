#!/bin/bash
# Tier-2 for fourc::solid_mechanics#0 — `KINEM: linear` on a hyperelastic
# material (MAT_ElastHyper + ELAST_CoupNeoHooke) is accepted without a word of
# complaint and returns a badly wrong deflection.
#
# One slender HEX8 cantilever, one Neo-Hookean material, one tip traction ramped
# over 8 load steps.  The only thing that differs between the two arms is the
# KINEM token on the element line.  The deck's RESULT DESCRIPTION is pinned to
# the KINEM: nonlinear answer, so the nonlinear arm exits 0 and the linear arm
# fails the same test and prints its own number next to it.
#
# The important half of the claim is the SILENCE: 4C emits no warning, no
# "linear kinematics with a hyperelastic law" note, nothing.  KINEM_WARNINGS=0
# and KINEM_MENTIONED_IN_LOG=0 pin that.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = KINEM token, $2 = output file
python3 - "$1" "$2" <<'PY'
import sys
kinem, out = sys.argv[1], sys.argv[2]
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
            eles.append(f'  - "{e} SOLID HEX8 {" ".join(str(x) for x in q)} MAT 1 KINEM {kinem}"')
clamp = [nid[(0, j, k)] for k in range(nz+1) for j in range(ny+1)]
tip = [nid[(nx, j, k)] for k in range(nz+1) for j in range(ny+1)]
open(out, "w").write(f"""PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.125
  NUMSTEP: 8
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 1.0
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 1000.0
      NUE: 0.3
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
    VAL: [0.0, 0.0, -1.5]
    FUNCT: [0, 0, 1]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:
{chr(10).join(f'  - "NODE {i} DSURFACE 1"' for i in clamp)}
{chr(10).join(f'  - "NODE {i} DSURFACE 2"' for i in tip)}
NODE COORDS:
{chr(10).join(coords)}
STRUCTURE ELEMENTS:
{chr(10).join(eles)}
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: {nid[(nx, 0, nz)]}
      QUANTITY: "dispz"
      VALUE: -5.75473889952457363
      TOLERANCE: 1.0e-08
""")
PY
}

deck "nonlinear" "$TMP/nonlinear.4C.yaml"
deck "linear"    "$TMP/linear.4C.yaml"

probe NONLINEAR "$TMP/nonlinear.4C.yaml"
probe LINEAR    "$TMP/linear.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/NONLINEAR.log"
grep -m1 -oE "dispz +at node +[0-9]+.*is WRONG --> actresult=[^,]*" "$TMP/LINEAR.log"
# 4C never says a word about the kinematics assumption not matching the material.
echo "KINEM_WARNINGS=$(grep -ciE 'warn' "$TMP/LINEAR.log")"
echo "KINEM_MENTIONED_IN_LOG=$(grep -ci 'kinem' "$TMP/LINEAR.log")"
exit 0
