#!/bin/bash
# Tier-2 for fourc::shell#2 — and a FALSIFICATION of its premise.
#
# The entry warned that "omitting THICK uses the default (often 1.0)".  There is
# no default: THICK is required:true on SHELL7P and leaving it out is a parse
# error that names the key.  The rest of the entry stands — the thickness drives
# the bending stiffness hard, so a wrong value is a wrong answer.
#
#   THICK_0p01  reference, pinned by the deck's result test -> exit 0
#   THICK_0p02  thickness doubled, nothing else changed     -> different answer
#   NO_THICK    key omitted                                 -> "Required value
#                                                              'THICK' not found"
#   THICKNESS   the MAT_Kirchhoff_Love_shell spelling       -> not a SHELL7P key
#
# The last arm matters because the two shell families in 4C disagree about where
# the thickness lives: SHELL7P takes THICK on the ELEMENT, the Kirchhoff-Love
# NURBS shell takes THICKNESS in the MATERIAL.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = thickness token(s) on the element line, $2 = out
python3 - "$1" "$2" <<'PY'
import sys
thick, out = sys.argv[1], sys.argv[2]
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
                    f'{thick}EAS N_4 N_4 N_4 none none SDC 1.0 USE_ANS true"')
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
      TOLERANCE: 1.0e-08
""")
PY
}

deck "THICK 0.01 "     "$TMP/t001.4C.yaml"
deck "THICK 0.02 "     "$TMP/t002.4C.yaml"
deck ""               "$TMP/nothick.4C.yaml"
deck "THICKNESS 0.01 " "$TMP/thickness.4C.yaml"

probe THICK_0p01 "$TMP/t001.4C.yaml"
probe THICK_0p02 "$TMP/t002.4C.yaml"
probe NO_THICK   "$TMP/nothick.4C.yaml"
probe THICKNESS  "$TMP/thickness.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/THICK_0p01.log"
grep -m1 -oE "is WRONG --> actresult=[^,]*" "$TMP/THICK_0p02.log"
# There is no default: the key is required and the message names it.
grep -m1 -F "Required value 'THICK' not found in input line" "$TMP/NO_THICK.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/NO_THICK.log"
# ...and the material-side spelling is not accepted on the element line.  Note
# WHICH key the diagnostic names: writing THICKNESS lets the parser consume the
# THICK prefix and then complain about the NEXT key, SDC.  The message points at
# a key that was never the problem.
grep -m1 -F "Required value 'SDC' not found in input line" "$TMP/THICKNESS.log"
echo "NO_THICK_STEPS=$(grep -c '^Finalised step' "$TMP/NO_THICK.log")"
exit 0
