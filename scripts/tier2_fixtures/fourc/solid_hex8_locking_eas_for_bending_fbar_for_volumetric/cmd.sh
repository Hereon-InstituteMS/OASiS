#!/bin/bash
# Tier-2 for fourc::solid_mechanics#3 — HEX8 locking, and which TECH cures which
# kind of it.  One slender inline cantilever (10 x 1 x 1 HEX8, L/h = 20), one tip
# traction, one static step.  The deck's RESULT DESCRIPTION is pinned to the
# TECH: eas_full answer, which is the one within 1% of the Euler-Bernoulli value
# for this beam, so the EAS arm exits 0 and every other arm fails the SAME test
# and prints its own number beside it.
#
#   PLAIN        no TECH key         -> bending-locked, far too stiff
#   EAS_FULL     TECH: eas_full      -> the reference answer, exit 0
#   FBAR         TECH: fbar          -> does NOT cure bending locking
#   NU0499_PLAIN NUE 0.499, no TECH  -> volumetric locking on top of it
#   NU0499_FBAR  NUE 0.499 + fbar    -> fbar DOES cure the volumetric part
#   BAD_TECH     TECH: eas           -> enum, not a free-form word
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra element-line text, $2 = NUE, $3 = output file
python3 - "$1" "$2" "$3" <<'PY'
import sys
tech, nue, out = sys.argv[1], sys.argv[2], sys.argv[3]
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
            eles.append(f'  - "{e} SOLID HEX8 {" ".join(str(x) for x in q)} '
                        f'MAT 1 KINEM nonlinear{tech}"')
clamp = [nid[(0, j, k)] for k in range(nz+1) for j in range(ny+1)]
tip = [nid[(nx, j, k)] for k in range(nz+1) for j in range(ny+1)]
open(out, "w").write(f"""PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: {nue}
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
    VAL: [0.0, 0.0, -1.0e-02]
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
      VALUE: -1.58679680112244814e-01
      TOLERANCE: 1.0e-09
""")
PY
}

deck ""                  0.3   "$TMP/plain.4C.yaml"
deck " TECH eas_full"    0.3   "$TMP/eas.4C.yaml"
deck " TECH fbar"        0.3   "$TMP/fbar.4C.yaml"
deck ""                  0.499 "$TMP/nu.4C.yaml"
deck " TECH fbar"        0.499 "$TMP/nufbar.4C.yaml"
deck " TECH eas"         0.3   "$TMP/badtech.4C.yaml"

probe EAS_FULL     "$TMP/eas.4C.yaml"
probe PLAIN        "$TMP/plain.4C.yaml"
probe FBAR         "$TMP/fbar.4C.yaml"
probe NU0499_PLAIN "$TMP/nu.4C.yaml"
probe NU0499_FBAR  "$TMP/nufbar.4C.yaml"
probe BAD_TECH     "$TMP/badtech.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/EAS_FULL.log"
for a in PLAIN FBAR NU0499_PLAIN NU0499_FBAR; do
  printf '%s ' "$a"
  grep -m1 -oE "is WRONG --> actresult=[^,]*" "$TMP/$a.log"
done
grep -m1 -F "Could not parse value 'eas' as an enum constant of type 'ElementTechnology'." "$TMP/BAD_TECH.log"
exit 0
