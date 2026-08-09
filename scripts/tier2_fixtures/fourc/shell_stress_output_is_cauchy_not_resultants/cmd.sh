#!/bin/bash
# Tier-2 for fourc::shell#4 — and a FALSIFICATION of its recipe.
#
# The entry told you to "output both N and M tensors via STRESS_STRAIN".  There
# is no N/M resultant output in 4C, and STRESS_STRAIN is not a SHELL7P element
# key:
#   * on the ELEMENT line it is rejected outright (it is a WALL / MEMBRANE key);
#   * as IO/RUNTIME VTK OUTPUT/STRUCTURE.STRESS_STRAIN (a bool) plus IO's
#     STRUCT_STRESS / STRUCT_STRAIN enums it works, and what lands in the .vtu
#     is the CAUCHY stress and GREEN-LAGRANGE strain tensors in global xyz —
#     element_cauchy_stresses_xyz, nodal_cauchy_stresses_xyz,
#     element_GL_strains_xyz, nodal_GL_strains_xyz.  No N_xx, no M_xx.
#   * the only shell-specific extra field is OPTIONAL_QUANTITY, whose
#     shell7pthicknessdirector value writes shell7p_thickness_director.
# Through-thickness resultants have to be integrated by the post-processor.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = extra element tokens, $2 = IO block, $3 = out
python3 - "$1" "$2" "$3" <<'PY'
import sys
extra, io, out = sys.argv[1], sys.argv[2], sys.argv[3]
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
{io}STRUCTURAL DYNAMIC:
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
""")
PY
}

IOBLOCK='IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
  OPTIONAL_QUANTITY: "shell7pthicknessdirector"
'

deck " STRESS_STRAIN plane_stress" ""         "$TMP/eleskey.4C.yaml"
deck ""                           "$IOBLOCK" "$TMP/vtk.4C.yaml"

probe ELEMENT_KEY "$TMP/eleskey.4C.yaml"
probe VTK_OUTPUT  "$TMP/vtk.4C.yaml"

# STRESS_STRAIN is not a SHELL7P element key.
grep -m1 -F "After parsing, the line still contains 'STRESS_STRAIN plane_stress'." "$TMP/ELEMENT_KEY.log"
grep -m1 -F "4C_io_input_spec.cpp" "$TMP/ELEMENT_KEY.log"
grep -m1 -F "processor 0 finished normally" "$TMP/VTK_OUTPUT.log"

VTU=$(find "$TMP" -name 'structure-*.vtu' | sort | tail -1)
[ -n "$VTU" ] || { echo "FIXTURE_ABORT=no_vtu_written"; exit 3; }
echo "VTU_FIELDS=$(grep -oE 'Name="[a-zA-Z0-9_]+"' "$VTU" | sort -u | tr '\n' ' ')"
echo "VTU_HAS_MEMBRANE_FORCE_RESULTANT=$(grep -ciE 'Name="(N_xx|n_xx|membrane_forces|stress_resultant)' "$VTU")"
echo "VTU_HAS_BENDING_MOMENT_RESULTANT=$(grep -ciE 'Name="(M_xx|m_xx|bending_moments|moment_resultant)' "$VTU")"
exit 0
