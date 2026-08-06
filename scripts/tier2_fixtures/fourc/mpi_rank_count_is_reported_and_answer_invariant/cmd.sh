#!/bin/bash
# Tier-2 for fourc::input_format#16 — MPI is 4C's domain decomposition, the
# standard invocation is `mpirun -np N 4C input out`, and forgetting mpirun
# leaves you on a single rank on a many-core box.
#
# The rank count is not something you have to infer from a stopwatch: 4C PRINTS
# it in its startup banner ("Total number of MPI ranks: N") and prints the
# number of ranks the redistribution used.  That is the observable here, along
# with the fact that the decomposition does not change the answer -- the 4x4x4
# HEX8 cantilever gives the identical displacement on 1 and on 2 ranks.
#
# OMP_NUM_THREADS is the thread-level knob and is orthogonal: libgomp is linked,
# and setting it changes neither the rank count nor the answer.
. "$(dirname "$0")/../_lib/preamble.sh"

command -v mpirun >/dev/null 2>&1 || { echo "FIXTURE_ABORT=no_mpirun"; exit 3; }

DECK="$TMP/mesh.4C.yaml"
python3 - "$DECK" <<'PY'
import sys
n = 4
idx = {}; k = 1; nodes = []
for iz in range(n + 1):
    for iy in range(n + 1):
        for ix in range(n + 1):
            idx[(ix, iy, iz)] = k
            nodes.append('  - "NODE %d COORD %.10f %.10f %.10f"' % (k, ix / n, iy / n, iz / n))
            k += 1
els = []; e = 1
for iz in range(n):
    for iy in range(n):
        for ix in range(n):
            c = [idx[(ix, iy, iz)], idx[(ix + 1, iy, iz)], idx[(ix + 1, iy + 1, iz)], idx[(ix, iy + 1, iz)],
                 idx[(ix, iy, iz + 1)], idx[(ix + 1, iy, iz + 1)], idx[(ix + 1, iy + 1, iz + 1)], idx[(ix, iy + 1, iz + 1)]]
            els.append('  - "%d SOLID HEX8 %s MAT 1 KINEM nonlinear"' % (e, " ".join(map(str, c))))
            e += 1
fix = ['  - "NODE %d DSURFACE 1"' % idx[(0, iy, iz)] for iz in range(n + 1) for iy in range(n + 1)]
load = ['  - "NODE %d DSURFACE 2"' % idx[(n, iy, iz)] for iz in range(n + 1) for iy in range(n + 1)]
probe = idx[(n, n, n)]
open(sys.argv[1], "w").write("""PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 1
  MAXTIME: 0.1
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 1, 0]
    FUNCT: [0, 0, 0]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:
""" + "\n".join(fix + load) + """
NODE COORDS:
""" + "\n".join(nodes) + """
STRUCTURE ELEMENTS:
""" + "\n".join(els) + """
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: %d
      QUANTITY: "dispy"
      VALUE: 6.40632246176932405e-03
      TOLERANCE: 1e-12
""" % probe)
PY

# (a) plain invocation, which is what you get when you forget mpirun
probe SERIAL "$DECK"
# (b) the standard invocation
stdbuf -oL -eL mpirun -np 2 --oversubscribe "$BIN" "$DECK" "$TMP/o_par" > "$TMP/PAR.log" 2>&1
echo "EXIT_PAR=$?"
# (c) threads are a different knob and do not decompose anything
OMP_NUM_THREADS=4 probe OMP "$DECK"

grep -m1 -F "Total number of MPI ranks: 1" "$TMP/SERIAL.log"
grep -m1 -F "Total number of MPI ranks: 2" "$TMP/PAR.log"
grep -m1 -F "Number of procs used for redistribution: 1" "$TMP/SERIAL.log"
grep -m1 -F "Number of procs used for redistribution: 2" "$TMP/PAR.log"
grep -m1 -F "processor 0 finished normally" "$TMP/PAR.log"
echo "SERIAL_CORRECT=$(grep -c 'is CORRECT' "$TMP/SERIAL.log")"
echo "PAR_CORRECT=$(grep -c 'is CORRECT' "$TMP/PAR.log")"
echo "DECOMPOSITION_CHANGED_THE_ANSWER=$([ "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/SERIAL.log")" = "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/PAR.log")" ] && echo no || echo yes)"
# Threads are not ranks.
grep -m1 -F "Total number of MPI ranks: 1" "$TMP/OMP.log"
echo "OMP_CHANGED_THE_ANSWER=$([ "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/SERIAL.log")" = "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/OMP.log")" ] && echo no || echo yes)"
LIB=$(dirname "$BIN")/lib4C.so
[ -f "$LIB" ] || LIB="$BIN"
echo "OPENMP_RUNTIME_LINKED=$(ldd "$LIB" 2>/dev/null | grep -c 'libgomp')"
exit 0
