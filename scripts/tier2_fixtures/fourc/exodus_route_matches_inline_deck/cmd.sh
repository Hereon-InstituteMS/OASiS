#!/bin/bash
# Tier-2 for fourc::input_format#12 — the ExodusII route and the inline route
# are exactly interchangeable, and the reason to prefer the mesh file is DECK
# SIZE, not a parse-time cliff.
#
# THE CLAIM'S NUMBER DOES NOT REPRODUCE.  The entry says an inline YAML with
# more than 1000 lines "takes 30+ seconds to parse".  The 12x12x12 cantilever
# built here is 4306 lines and 2197 nodes -- squarely in the claimed regime --
# and 4C's own INPUT phase for it is a fraction of a second.  The elapsed
# figures are PRINTED below for the reader and are deliberately not asserted;
# what the fixture stands on is structural: the same mesh costs 4306 lines
# inline against 388 lines of YAML plus one .e file, and BOTH ROUTES REACH THE
# SAME ANSWER, pinned to 1e-12 in each deck's own RESULT DESCRIPTION.
#
# So the guidance survives with a different justification: use a mesh file
# because the deck stays readable and reviewable, not because YAML is slow.
. "$(dirname "$0")/../_lib/preamble.sh"

python3 -c "import meshio, netCDF4" 2>/dev/null || { echo "FIXTURE_ABORT=no_meshio_or_netcdf4"; exit 3; }
cd "$TMP" || exit 3

python3 - <<'PY'
import numpy as np, meshio, netCDF4
N = 12                       # 1728 HEX8, 2197 nodes
idx = {}; k = 1; nodes = []; pts = []
for iz in range(N + 1):
    for iy in range(N + 1):
        for ix in range(N + 1):
            idx[(ix, iy, iz)] = k
            nodes.append('  - "NODE %d COORD %.10f %.10f %.10f"' % (k, ix / N, iy / N, iz / N))
            pts.append([ix / N, iy / N, iz / N]); k += 1
els = []; conn = []; e = 1
for iz in range(N):
    for iy in range(N):
        for ix in range(N):
            c = [idx[(ix, iy, iz)], idx[(ix + 1, iy, iz)], idx[(ix + 1, iy + 1, iz)], idx[(ix, iy + 1, iz)],
                 idx[(ix, iy, iz + 1)], idx[(ix + 1, iy, iz + 1)], idx[(ix + 1, iy + 1, iz + 1)], idx[(ix, iy + 1, iz + 1)]]
            els.append('  - "%d SOLID HEX8 %s MAT 1 KINEM nonlinear"' % (e, " ".join(map(str, c))))
            conn.append([x - 1 for x in c]); e += 1
fix  = ['  - "NODE %d DSURFACE 1"' % idx[(0, iy, iz)] for iz in range(N + 1) for iy in range(N + 1)]
load = ['  - "NODE %d DSURFACE 2"' % idx[(N, iy, iz)] for iz in range(N + 1) for iy in range(N + 1)]
probe_node = idx[(N, N, N)]

HEAD = '''PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
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
    VAL: [0, 0.02, 0]
    FUNCT: [0, 0, 0]
    TYPE: "Live"
'''
TAIL = '''MATERIALS:
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
      VALUE: 1.38646642128815859e-04
      TOLERANCE: 1e-12
''' % probe_node
GEO = '''STRUCTURE GEOMETRY:
  FILE: "mesh.e"
  SHOW_INFO: "detailed_summary"
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: "nonlinear"
'''
open("inline.yaml", "w").write(
    HEAD + "DSURF-NODE TOPOLOGY:\n" + "\n".join(fix + load) +
    "\nNODE COORDS:\n" + "\n".join(nodes) +
    "\nSTRUCTURE ELEMENTS:\n" + "\n".join(els) + "\n" + TAIL)
open("exo.yaml", "w").write(
    HEAD + GEO + "DSURF-NODE TOPOLOGY:\n" + "\n".join(fix + load) + "\n" + TAIL)
meshio.write("mesh.e", meshio.Mesh(np.array(pts), [("hexahedron", np.array(conn))]))
ds = netCDF4.Dataset("mesh.e", "r+"); ds.variables["eb_prop1"][:] += 1; ds.close()
print("MESH_NODES=%d" % len(nodes))
print("MESH_ELEMENTS=%d" % len(els))
PY

echo "INLINE_DECK_LINES=$(wc -l < inline.yaml)"
echo "EXODUS_DECK_LINES=$(wc -l < exo.yaml)"
echo "INLINE_OVER_1000_LINES=$([ "$(wc -l < inline.yaml)" -gt 1000 ] && echo yes || echo no)"
echo "DECK_LINE_RATIO_OVER_5=$([ $(( $(wc -l < inline.yaml) / $(wc -l < exo.yaml) )) -ge 5 ] && echo yes || echo no)"

probe INLINE inline.yaml
probe EXODUS exo.yaml

grep -m1 -F "processor 0 finished normally" "$TMP/INLINE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/EXODUS.log"
echo "INLINE_CORRECT=$(grep -c 'is CORRECT' "$TMP/INLINE.log")"
echo "EXODUS_CORRECT=$(grep -c 'is CORRECT' "$TMP/EXODUS.log")"
echo "EXODUS_READ_THE_MESH=$(grep -c 'cell-block 1 (): 1728 cells of type hex8' "$TMP/EXODUS.log")"

# Reported for the reader only.  Never asserted: it depends on the box.
echo "(informational) inline input phase: $(grep -m1 'Total wall time for INPUT' "$TMP/INLINE.log")"
echo "(informational) exodus input phase: $(grep -m1 'Total wall time for INPUT' "$TMP/EXODUS.log")"
exit 0
