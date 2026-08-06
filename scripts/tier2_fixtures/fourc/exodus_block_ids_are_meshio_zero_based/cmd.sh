#!/bin/bash
# Tier-2 for fourc::input_format#0 — meshio really does write ExodusII element
# block IDs starting at 0, 4C's ELEMENT_BLOCKS really are matched by that id,
# and the mismatch is NOT diagnosed.
#
# What actually happens is worse than a cryptic message: 4C prints the mesh
# summary it read ("cell-block 0 (): 1 cells of type hex8"), accepts an
# ELEMENT_BLOCKS entry for ID 1 without a word, builds a discretisation with no
# elements in it, and dies inside Amesos2/UMFPACK with an unhandled
# std::runtime_error and shell status 134.  There is NO `PROC 0 ERROR` block, so
# the usual way of reading a 4C failure finds nothing at all.
#
# The claimed 'Pressure map empty' signal appears nowhere.
#
# The recorded fix is executed too: netCDF4, eb_prop1 += 1, and the same deck
# then runs and reproduces the answer of the ID-0 arm exactly.
. "$(dirname "$0")/../_lib/preamble.sh"

python3 -c "import meshio, netCDF4" 2>/dev/null || { echo "FIXTURE_ABORT=no_meshio_or_netcdf4"; exit 3; }

cd "$TMP" || exit 3
python3 - <<'PY'
import shutil
import numpy as np, meshio, netCDF4
pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
mesh = meshio.Mesh(pts, [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))])
meshio.write("m_meshio.e", mesh)          # straight out of meshio
shutil.copy("m_meshio.e", "m_patched.e")
ds = netCDF4.Dataset("m_patched.e", "r+")  # the recorded fix
ds.variables["eb_prop1"][:] += 1
ds.close()
for f in ("m_meshio.e", "m_patched.e"):
    ds = netCDF4.Dataset(f)
    print("EB_PROP1_%s=%s" % (f.split(".")[0], list(ds.variables["eb_prop1"][:])))
    ds.close()
PY

mk() {  # $1 = ELEMENT_BLOCKS id, $2 = mesh file, $3 = out file
cat > "$3" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURE GEOMETRY:
  FILE: "$2"
  SHOW_INFO: "detailed_summary"
  ELEMENT_BLOCKS:
    - ID: $1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: "nonlinear"
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
  - "NODE 1 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 1"
  - "NODE 8 DSURFACE 1"
  - "NODE 2 DSURFACE 2"
  - "NODE 3 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: 4.47909266337460053e-03
      TOLERANCE: 1e-12
YAML
}

mk 1 m_meshio.e  "$TMP/id1_meshio.yaml"    # 4C convention against a meshio mesh
mk 0 m_meshio.e  "$TMP/id0_meshio.yaml"    # match what meshio actually wrote
mk 1 m_patched.e "$TMP/id1_patched.yaml"   # the netCDF4 fix

probe ID1MESHIO  "$TMP/id1_meshio.yaml"
probe ID0MESHIO  "$TMP/id0_meshio.yaml"
probe ID1PATCHED "$TMP/id1_patched.yaml"

# 4C reports the block numbering it read -- this line IS the diagnostic.
grep -m1 -F "cell-block 0 (): 1 cells of type hex8" "$TMP/ID1MESHIO.log"
grep -m1 -F "cell-block 1 (): 1 cells of type hex8" "$TMP/ID1PATCHED.log"
# The mismatched arm dies with no 4C diagnostic whatsoever.
echo "MISMATCH_FOURC_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/ID1MESHIO.log")"
echo "MISMATCH_MENTIONED_THE_BLOCK_ID=$(grep -ci 'block.*id\|ID: 1' "$TMP/ID1MESHIO.log")"
grep -m1 -F "terminate called after throwing an instance of 'std::runtime_error'" "$TMP/ID1MESHIO.log"
grep -m1 -F "umfpack_solve has error code: -3" "$TMP/ID1MESHIO.log"
echo "CLAIMED_PRESSURE_MAP_EMPTY_TEXT=$(grep -ci 'Pressure map empty' "$TMP/ID1MESHIO.log")"
# Both correct spellings reach the same answer.
grep -m1 -F "processor 0 finished normally" "$TMP/ID0MESHIO.log"
echo "ID0MESHIO_CORRECT=$(grep -c 'is CORRECT' "$TMP/ID0MESHIO.log")"
echo "ID1PATCHED_CORRECT=$(grep -c 'is CORRECT' "$TMP/ID1PATCHED.log")"
echo "FIX_REPRODUCES_THE_ANSWER=$([ "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/ID0MESHIO.log")" = "$(grep -o 'abs(diff)= [0-9.e+-]*' "$TMP/ID1PATCHED.log")" ] && echo yes || echo no)"
exit 0
