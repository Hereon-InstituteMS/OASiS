#!/bin/bash
# Tier-2 for fourc::membrane#1 — and a correction to its recipe.
#
# The entry said: "Apply prestress via PRESTRESS section or internal pressure
# DESIGN SURF NEUMANN."  On a FLAT sheet neither of those works, because both
# are loads and neither puts stress into the reference configuration:
#
#   PRESSURE_ONLY    flat patch, orthopressure, no edge drive  -> dies, SIGFPE
#   PRESTRESS_MULF   the same plus STRUCTURAL DYNAMIC PRESTRESS: "MULF"
#                                                              -> dies, SIGFPE
#   INPLANE_TENSION  the same mesh with the edges stretched    -> 4 steps, exit 0
#   CURVED_PRESSURE  upstream membrane_cyl_new_struc, driven by orthopressure
#                    alone on a CURVED surface                 -> runs, exit 0
#
# What stabilises a membrane is IN-PLANE STRESS.  Pressure produces it only once
# the surface has curvature to react against; on a flat sheet it produces none,
# and the out-of-plane block of the tangent stays empty.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = edge drive scale, $2 = pressure, $3 = extra STRUCTURAL DYNAMIC, $4 = out
python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
scale, press, extra, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
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
STRUCTURAL DYNAMIC:{extra}
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
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 0, 0]
    VAL: [{press}, 0.0, 0.0]
    FUNCT: [2, 0, 0]
    TYPE: "orthopressure"
DLINE-NODE TOPOLOGY:
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

deck 0.0 0.05 ""                                              "$TMP/pressure.4C.yaml"
deck 0.0 0.05 $'\n  PRESTRESS: "MULF"\n  PRESTRESSTIME: 0.5'  "$TMP/mulf.4C.yaml"
deck 0.1 0.00 ""                                              "$TMP/tension.4C.yaml"

# These two die on SIGFPE; the shell's job message is locale dependent.
( probe PRESSURE_ONLY  "$TMP/pressure.4C.yaml" ) 2>/dev/null
( probe PRESTRESS_MULF "$TMP/mulf.4C.yaml" )     2>/dev/null
probe INPLANE_TENSION "$TMP/tension.4C.yaml"

CURVED=$(upstream membrane_cyl_new_struc.4C.yaml) || exit 3
grep -q 'TYPE: "orthopressure"' "$CURVED" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
cp "$CURVED" "$TMP/curved.4C.yaml"
probe CURVED_PRESSURE "$TMP/curved.4C.yaml"

echo "PRESSURE_ONLY_STEPS=$(grep -c '^Finalised step' "$TMP/PRESSURE_ONLY.log")"
echo "PRESTRESS_MULF_STEPS=$(grep -c '^Finalised step' "$TMP/PRESTRESS_MULF.log")"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/PRESSURE_ONLY.log"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/PRESTRESS_MULF.log"
# In-plane tension is what works, on the identical mesh.
grep -m1 -F "processor 0 finished normally" "$TMP/INPLANE_TENSION.log"
echo "INPLANE_TENSION_STEPS=$(grep -c '^Finalised step' "$TMP/INPLANE_TENSION.log")"
# And pressure DOES stabilise once the surface is curved.
echo "CURVED_PRESSURE_STEPS=$(grep -c '^Finalised step' "$TMP/CURVED_PRESSURE.log")"
exit 0
