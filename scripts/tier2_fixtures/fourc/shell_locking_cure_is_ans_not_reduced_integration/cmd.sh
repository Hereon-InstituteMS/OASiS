#!/bin/bash
# Tier-2 for fourc::shell#1 — Reissner-Mindlin shells do lock, but the cure in
# 4C is NOT "reduced integration": SHELL7P has no integration-rule key at all.
# The two knobs it does have are EAS (a 5-slot vector) and USE_ANS (assumed
# natural strain, which is the transverse-shear cure).
#
# Flat cantilever plate, 8 x 2 SHELL7P QUAD4, t/L = 1e-3.  The deck's result
# test is pinned to the EAS + ANS answer.
#
#   EAS_ANS      EAS N_4 N_4 N_4 none none, USE_ANS true  -> exit 0
#   EAS_NO_ANS   same EAS, USE_ANS false                  -> shear-locked, three
#                                                            orders of magnitude
#                                                            too stiff
#   NO_EAS       EAS none x5, USE_ANS true                -> reference BLAS
#                                                            aborts the process
#                                                            with EXIT 0 and zero
#                                                            steps: a silent
#                                                            no-op that looks
#                                                            like success
#   GP_KEY       the WALL-style "GP 2 2" integration key   -> not a SHELL7P key
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = EAS vector, $2 = USE_ANS, $3 = extra tokens, $4 = out
python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
eas, ans, extra, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
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
                    f'THICK 0.01 EAS {eas} SDC 1.0 USE_ANS {ans}{extra}"')
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

deck "N_4 N_4 N_4 none none"     true  ""        "$TMP/easans.4C.yaml"
deck "N_4 N_4 N_4 none none"     false ""        "$TMP/eas.4C.yaml"
deck "none none none none none"  true  ""        "$TMP/noeas.4C.yaml"
deck "N_4 N_4 N_4 none none"     true  " GP 2 2" "$TMP/gp.4C.yaml"

probe EAS_ANS    "$TMP/easans.4C.yaml"
probe EAS_NO_ANS "$TMP/eas.4C.yaml"
probe NO_EAS     "$TMP/noeas.4C.yaml"
probe GP_KEY     "$TMP/gp.4C.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/EAS_ANS.log"
# Switching ANS off on the same element locks it solid.
grep -m1 -oE "is WRONG --> actresult=[^,]*" "$TMP/EAS_NO_ANS.log"
# EAS switched off entirely: reference BLAS calls XERBLA, which STOPs the
# process with status 0.  No steps, no result test, and an exit code that
# reads as success.
echo "NO_EAS_STEPS=$(grep -c '^Finalised step' "$TMP/NO_EAS.log")"
echo "NO_EAS_RESULT_TESTS_RUN=$(grep -c 'Checking results of' "$TMP/NO_EAS.log")"
grep -m1 -F "On entry to DGEMM parameter number 10 had an illegal value" "$TMP/NO_EAS.log"
# There is no integration-rule key on this element; the WALL-style one is
# rejected, and the message lists every key SHELL7P does accept.
grep -m1 -F "After parsing, the line still contains 'GP 2 2'." "$TMP/GP_KEY.log"
grep -m1 -F "Parsed parameters: AXI : none CIR : none EAS :" "$TMP/GP_KEY.log"
exit 0
