#!/bin/bash
# Tier-2 for fourc::structural_dynamics#7 — the SIGN of DENS decides where the
# run dies, and the nasty case is the one that looks like a solver problem.
#
# MAT_Struct_StVenantKirchhoff puts no validator on DENS, so a leading minus
# survives the parser untouched.  Same slender transient cantilever, three
# densities:
#
#   DENS_POS   +1.0 -> all 12 steps, exit 0
#   DENS_ZERO   0.0 -> 0 steps, singular-matrix throw from the integrator
#   DENS_NEG   -1.0 -> several steps finalise, THEN a generic NOX
#                      "The nonlinear solver did not converge!" that says
#                      nothing about the material at all
#
# For a reader of the log: a mid-run NOX non-convergence in an otherwise
# well-posed deck is a reason to check the SIGN of DENS before touching TOLRES
# or MAXITER.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = DENS, $2 = out
python3 - "$1" "$2" <<'PY'
import sys
dens, out = sys.argv[1], sys.argv[2]
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
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "GenAlpha"
  TIMESTEP: 0.05
  NUMSTEP: 12
  MAXTIME: 0.6
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
      DENS: {dens}
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

deck  1.0 "$TMP/pos.4C.yaml"
deck  0.0 "$TMP/zero.4C.yaml"
deck -1.0 "$TMP/neg.4C.yaml"

probe DENS_POS  "$TMP/pos.4C.yaml"
probe DENS_ZERO "$TMP/zero.4C.yaml"
probe DENS_NEG  "$TMP/neg.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/DENS_POS.log"
echo "DENS_POS_STEPS=$(grep -c '^Finalised step' "$TMP/DENS_POS.log")"
echo "DENS_ZERO_STEPS=$(grep -c '^Finalised step' "$TMP/DENS_ZERO.log")"
grep -m1 -F "You are about to invert a singular matrix!" "$TMP/DENS_ZERO.log"
# The negative density parses, integrates several steps, and only then dies.
echo "DENS_NEG_STEPS=$(grep -c '^Finalised step' "$TMP/DENS_NEG.log")"
echo "DENS_NEG_DIED_AFTER_RUNNING=$([ "$(grep -c '^Finalised step' "$TMP/DENS_NEG.log")" -gt 0 ] && echo yes || echo no)"
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/DENS_NEG.log"
grep -m1 -F "4C_solver_nonlin_nox_problem.cpp" "$TMP/DENS_NEG.log"
# ...and that diagnostic mentions neither the density nor the material.
echo "DENS_NEG_DIAGNOSTIC_NAMES_DENSITY=$(grep -ci 'densit' "$TMP/DENS_NEG.log")"
exit 0
