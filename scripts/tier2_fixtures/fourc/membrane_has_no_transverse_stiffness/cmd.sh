#!/bin/bash
# Tier-2 for fourc::membrane#0 — "membranes have ZERO bending stiffness" is not
# a modelling nicety in 4C, it decides whether the deck runs at all.
#
# One flat 4x4 MEMBRANE4 patch in the z = 0 plane, all four edges Dirichlet.
# Three arms:
#
#   TRANSVERSE_LOAD    a normal ("Live", z-slot) surface traction — the load you
#                      would put on a plate — is refused outright: a MEMBRANE
#                      surface load is only accepted on the FIRST dof slot
#   NO_INPLANE_STRESS  edges held at zero, no load: the out-of-plane block of the
#                      tangent is identically empty because there is no bending
#                      term to fill it, and the run dies on a floating-point
#                      exception inside UMFPACK's triangular solve
#   INPLANE_TENSION    the SAME mesh with the edges pulled outward runs all four
#                      steps: the geometric stiffness from the in-plane stress is
#                      the only thing holding the sheet out of plane
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = edge drive scale, $2 = neumann block, $3 = out
python3 - "$1" "$2" "$3" <<'PY'
import sys
scale, neumann, out = sys.argv[1], sys.argv[2], sys.argv[3]
n, L = 4, 1.0
nid, coords = {}, []
c = 0
for j in range(n + 1):
    for i in range(n + 1):
        c += 1
        nid[(i, j)] = c
        coords.append(f'  - "NODE {c} COORD {i*L/n:.16e} {j*L/n:.16e} 0.0"')
eles = []
for j in range(n):
    for i in range(n):
        e = j * n + i + 1
        q = [nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]]
        eles.append(f'  - "{e} MEMBRANE4 QUAD4 {q[0]} {q[1]} {q[2]} {q[3]} MAT 1 '
                    f'KINEM nonlinear THICK 0.01 STRESS_STRAIN plane_stress"')
edge = sorted({nid[(i, j)] for i in range(n+1) for j in range(n+1)
               if i in (0, n) or j in (0, n)})
allsurf = sorted(nid.values())
open(out, "w").write(f"""PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.25
  NUMSTEP: 4
  MAXTIME: 1.0
  TOLDISP: 1.0e-09
  TOLRES: 1.0e-08
  MAXITER: 25
  LOADLIN: true
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_Membrane_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 1.0
  - MAT: 2
    ELAST_IsoNeoHooke:
      MUE:
        constant: 40.0
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "x*t"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "y*t"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [{scale}, {scale}, 0.0]
    FUNCT: [1, 1, 0]
{neumann}DLINE-NODE TOPOLOGY:
{chr(10).join(f'  - "NODE {i} DLINE 1"' for i in edge)}
DSURF-NODE TOPOLOGY:
{chr(10).join(f'  - "NODE {i} DSURFACE 1"' for i in allsurf)}
NODE COORDS:
{chr(10).join(coords)}
STRUCTURE ELEMENTS:
{chr(10).join(eles)}
""")
PY
}

# A plate-style normal traction: third dof slot, TYPE Live.
NORMAL_TRACTION='DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.02]
    FUNCT: [0, 0, 2]
    TYPE: "Live"
'

deck 0.0 "$NORMAL_TRACTION" "$TMP/transverse.4C.yaml"
deck 0.0 ""                 "$TMP/nostress.4C.yaml"
deck 0.1 ""                 "$TMP/tension.4C.yaml"

probe TRANSVERSE_LOAD "$TMP/transverse.4C.yaml"
# This arm dies on SIGFPE; the shell's job message is locale dependent.
( probe NO_INPLANE_STRESS "$TMP/nostress.4C.yaml" ) 2>/dev/null
probe INPLANE_TENSION "$TMP/tension.4C.yaml"

# A membrane will not take a transverse surface load at all.
grep -m1 -F "membrane pressure on 1st dof only!" "$TMP/TRANSVERSE_LOAD.log"
grep -m1 -F "4C_membrane_evaluate.cpp" "$TMP/TRANSVERSE_LOAD.log"
# With no in-plane stress there is nothing in the out-of-plane block: the
# factorisation divides by zero and no 4C-level diagnostic is produced.
echo "NO_INPLANE_STRESS_STEPS=$(grep -c '^Finalised step' "$TMP/NO_INPLANE_STRESS.log")"
echo "NO_INPLANE_STRESS_4C_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/NO_INPLANE_STRESS.log")"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/NO_INPLANE_STRESS.log"
grep -m1 -oF "umfpack" "$TMP/NO_INPLANE_STRESS.log"
# The same mesh with in-plane tension is perfectly solvable.
grep -m1 -F "processor 0 finished normally" "$TMP/INPLANE_TENSION.log"
echo "INPLANE_TENSION_STEPS=$(grep -c '^Finalised step' "$TMP/INPLANE_TENSION.log")"
exit 0
