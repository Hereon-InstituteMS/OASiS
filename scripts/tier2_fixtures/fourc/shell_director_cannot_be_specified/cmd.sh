#!/bin/bash
# Tier-2 for fourc::shell#3 — and a FALSIFICATION of it.
#
# The entry said "Director vector must be SPECIFIED or auto-computed from
# element normal", and advised using smoothed nodal directors on curved meshes.
# There is no way to specify one: SHELL7P has no director key.  Its whole
# accepted key set is MAT / THICK / EAS / SDC / USE_ANS plus the optional
# orientation vectors RAD, AXI, CIR, FIBER1..3 — and 4C prints that list back at
# you when it rejects an unknown token.
#
#   BASE    no orientation vectors        -> exit 0, pinned by the result test
#   DIR     "DIR 0 0 1" appended          -> rejected, with the full key list
#   FIBER1  "FIBER1 1 0 0" appended       -> ACCEPTED, and the answer is
#                                            identical to BASE: the vectors that
#                                            do exist are material-orientation
#                                            data, not directors, and they are
#                                            inert for an isotropic material
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra element-line tokens, $2 = out
python3 - "$1" "$2" <<'PY'
import sys
extra, out = sys.argv[1], sys.argv[2]
nx, ny, L, b = 8, 2, 10.0, 2.0
nid, coords = {}, []
c = 0
for j in range(ny + 1):
    for i in range(nx + 1):
        c += 1
        nid[(i, j)] = c
        coords.append(f'  - "NODE {c} COORD {i*L/nx:.16e} {j*b/ny:.16e} 0.0"')
eles = []
for j in range(ny):
    for i in range(nx):
        e = j * nx + i + 1
        q = [nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]]
        eles.append(f'  - "{e} SHELL7P QUAD4 {q[0]} {q[1]} {q[2]} {q[3]} MAT 1 '
                    f'THICK 0.01 EAS N_4 N_4 N_4 none none SDC 1.0 USE_ANS true{extra}"')
clamp = [nid[(0, j)] for j in range(ny + 1)]
tip = [nid[(nx, j)] for j in range(ny + 1)]
open(out, "w").write(f"""PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 40
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1.0e+06
      NUE: 0.0
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [0, 0, 1, 0, 0, 0]
    VAL: [0, 0, -1.0e-03, 0, 0, 0]
    FUNCT: [0, 0, 1, 0, 0, 0]
DLINE-NODE TOPOLOGY:
{chr(10).join(f'  - "NODE {i} DLINE 1"' for i in clamp)}
{chr(10).join(f'  - "NODE {i} DLINE 2"' for i in tip)}
NODE COORDS:
{chr(10).join(coords)}
STRUCTURE ELEMENTS:
{chr(10).join(eles)}
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: {nid[(nx, 0)]}
      QUANTITY: "dispz"
      VALUE: -2.52367752125895084e+00
      TOLERANCE: 1.0e-12
""")
PY
}

deck ""               "$TMP/base.4C.yaml"
deck " DIR 0 0 1"     "$TMP/dir.4C.yaml"
deck " FIBER1 1 0 0"  "$TMP/fiber.4C.yaml"

probe BASE   "$TMP/base.4C.yaml"
probe DIR    "$TMP/dir.4C.yaml"
probe FIBER1 "$TMP/fiber.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
# There is no director key, and 4C lists everything SHELL7P does take.
grep -m1 -F "After parsing, the line still contains 'DIR 0 0 1'." "$TMP/DIR.log"
grep -m1 -F "Parsed parameters: AXI : none CIR : none EAS : N_4 N_4 N_4 none none FIBER1 : none FIBER2 : none FIBER3 : none MAT : 1 RAD : none SDC : 1 THICK : 0.01 USE_ANS : 1" "$TMP/DIR.log"
grep -m1 -F "4C_io_input_spec.cpp" "$TMP/DIR.log"
echo "DIRECTOR_KEY_IN_ACCEPTED_LIST=$(grep -c 'DIRECTOR' "$TMP/DIR.log")"
# The orientation vectors that DO exist change nothing on an isotropic material:
# the same result test, pinned to 1e-12, still passes.
echo "FIBER1_RESULT_FAILURES=$(grep -c 'is WRONG --> actresult=' "$TMP/FIBER1.log")"
exit 0
