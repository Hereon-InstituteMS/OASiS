#!/bin/bash
# Tier-2 for fourc::contact#14 — 'OUTPUT_CONTACT: true' writes contact tractions
# only if the PARENT section 'IO/RUNTIME VTK OUTPUT' with INTERVAL_STEPS is
# present as well, and forgetting the parent produces no error at all.
#
# Four runs of the same converging two-block mortar penalty deck, differing only
# in their IO sections:
#
#   parent + child(OUTPUT_CONTACT)  -> <prefix>-vtk-files/ with TWO series:
#                                      structure-* and structure-contact-*
#   child only, no parent           -> NO directory, no file, no message
#   parent + child, no OUTPUT_CONTACT -> only the structure-* series
#   neither section                 -> NO directory
#
# All four exit 0.  There is no error and no warning either way, so the only
# detector is the absence of the directory — which is what this fixture asserts.
# The contact series really does carry the tractions: the vtu point/cell arrays
# are read back and named below.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = IO block (may be empty), $2 = out
cat > "$2" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
${1}SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
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
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, -0.3]
    FUNCT: [0, 0, 1]
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
  - "NODE 8 DSURFACE 2"
  - "NODE 9 DSURFACE 3"
  - "NODE 10 DSURFACE 3"
  - "NODE 11 DSURFACE 3"
  - "NODE 12 DSURFACE 3"
  - "NODE 13 DSURFACE 4"
  - "NODE 14 DSURFACE 4"
  - "NODE 15 DSURFACE 4"
  - "NODE 16 DSURFACE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
  - "NODE 9 COORD 0.0 0.0 1.1"
  - "NODE 10 COORD 1.0 0.0 1.1"
  - "NODE 11 COORD 1.0 1.0 1.1"
  - "NODE 12 COORD 0.0 1.0 1.1"
  - "NODE 13 COORD 0.0 0.0 2.1"
  - "NODE 14 COORD 1.0 0.0 2.1"
  - "NODE 15 COORD 1.0 1.0 2.1"
  - "NODE 16 COORD 0.0 1.0 2.1"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
  - "2 SOLID HEX8 9 10 11 12 13 14 15 16 MAT 1 KINEM nonlinear"
YAML
}

BOTH='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  OUTPUT_CONTACT: true
'
CHILD_ONLY='IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  OUTPUT_CONTACT: true
'
NO_CONTACT='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
'

deck "$BOTH"       "$TMP/both.yaml"
deck "$CHILD_ONLY" "$TMP/child_only.yaml"
deck "$NO_CONTACT" "$TMP/no_contact.yaml"
deck ""            "$TMP/no_io.yaml"

probe BOTH_SECTIONS  "$TMP/both.yaml"
probe CHILD_ONLY     "$TMP/child_only.yaml"
probe NO_OUTPUT_CONTACT "$TMP/no_contact.yaml"
probe NO_IO_SECTIONS "$TMP/no_io.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/CHILD_ONLY.log"

for a in BOTH_SECTIONS CHILD_ONLY NO_OUTPUT_CONTACT NO_IO_SECTIONS; do
  d="$TMP/o_$a-vtk-files"
  echo "VTK_DIR_$a=$([ -d "$d" ] && echo yes || echo no)"
  echo "STRUCTURE_SERIES_$a=$(ls "$d" 2>/dev/null | grep -c '^structure-[0-9]')"
  echo "CONTACT_SERIES_$a=$(ls "$d" 2>/dev/null | grep -c '^structure-contact-')"
  echo "ERROR_BLOCKS_$a=$(grep -c 'PROC 0 ERROR' "$TMP/$a.log")"
done

# The contact series really is the tractions, read back out of the file.
python3 - "$TMP/o_BOTH_SECTIONS-vtk-files" <<'PY'
import glob, os, re, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "structure-contact-*-0.vtu")))
names = set()
for f in files[:3]:
    blob = open(f, "rb").read().decode("utf-8", "replace")
    names |= set(re.findall(r'Name="([A-Za-z0-9_]+)"', blob))
for want in ("norcontactstress", "tancontactstress", "gap", "activeset"):
    print("CONTACT_VTU_HAS_%s=%s" % (want, "yes" if want in names else "no"))
PY
exit 0
