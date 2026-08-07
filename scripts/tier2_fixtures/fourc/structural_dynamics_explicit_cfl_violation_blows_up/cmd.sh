#!/bin/bash
# Tier-2 for fourc::structural_dynamics#1 — explicit time integration is only
# conditionally stable, and 4C does not check the CFL condition for you.
#
# Same inline HEX8 cantilever, E = 1000, rho = 1, so the wave speed is
# c = sqrt(E/rho) ~ 31.6 and the smallest element edge is 0.5: the stability
# limit is around h/c ~ 1.6e-2.
#
#   STABLE    ExplicitEuler, dt = 1e-4  -> 8 steps, ||dx|| grows linearly
#   UNSTABLE  ExplicitEuler, dt = 5e-2  -> ||dx|| grows by orders of magnitude
#                                          per step and the process dies on a
#                                          floating-point exception
#   TYPO      DYNAMICTYPE "ExplEuler"   -> rejected at parse: the spelling is
#                                          "ExplicitEuler"
#
# 4C prints the per-step increment norm itself ("||dx||=..."), so the runaway is
# quotable straight out of the log without any post-processing.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = DYNAMICTYPE, $2 = TIMESTEP, $3 = out
python3 - "$1" "$2" "$3" <<'PY'
import sys
dyn, dt, out = sys.argv[1], sys.argv[2], sys.argv[3]
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
  DYNAMICTYPE: "{dyn}"
  TIMESTEP: {dt}
  NUMSTEP: 8
  MAXTIME: 1.0e+06
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

deck ExplicitEuler 1.0e-4 "$TMP/stable.4C.yaml"
deck ExplicitEuler 5.0e-2 "$TMP/unstable.4C.yaml"
deck ExplEuler     1.0e-4 "$TMP/typo.4C.yaml"

probe STABLE "$TMP/stable.4C.yaml"
# The unstable arm dies on SIGFPE; the shell's own job message is locale
# dependent, so keep it out of the fixture's stdout.
( probe UNSTABLE "$TMP/unstable.4C.yaml" ) 2>/dev/null
probe TYPO "$TMP/typo.4C.yaml"

echo "STABLE_STEPS=$(grep -c '^Finalised step' "$TMP/STABLE.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/STABLE.log"
# 4C prints the increment norm per step.  Under CFL it creeps; over it, it runs away.
echo "STABLE_LAST_DX=$(grep -oE '\|\|dx\|\|=[0-9.e+-]+' "$TMP/STABLE.log" | tail -1)"
echo "UNSTABLE_LAST_DX=$(grep -oE '\|\|dx\|\|=[0-9.e+-]+' "$TMP/UNSTABLE.log" | tail -1)"
echo "UNSTABLE_STEPS=$(grep -c '^Finalised step' "$TMP/UNSTABLE.log")"
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/UNSTABLE.log"
# Neither arm is warned about: 4C never mentions CFL or stability.
echo "CFL_MENTIONED=$(grep -ciE '\\bCFL\\b|courant|stability limit|time step too large' "$TMP/UNSTABLE.log")"
# And the scheme name is spelled out in full.
grep -m1 -F "Could not match this input" "$TMP/TYPO.log"
grep -m1 -F "ExplEuler" "$TMP/TYPO.log"
exit 0
