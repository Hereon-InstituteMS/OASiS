#!/bin/bash
# Tier-2 for fourc::xfem_fluid#6 -- an XFEM fluid writes no VTU files, per
# sub-domain or otherwise.
#
# Claimed: visualize('list') shows fluid_subdomain_0-*.vtu and
#          fluid_subdomain_1-*.vtu instead of a single fluid-*.vtu.
# Observed: with IO/RUNTIME VTK OUTPUT + IO/RUNTIME VTK OUTPUT/FLUID enabled,
#          a plain Fluid problem writes fluid-00001-0.vtu and a .pvd; the
#          identical configuration on a Fluid_XFEM problem writes ZERO .vtu and
#          ZERO .pvd, exits 0, and says nothing.  XFEM fluid output goes to the
#          legacy Ensight .result file and, if OUTPUT_GMSH is on, to Gmsh .pos
#          files.  There is no per-sub-domain VTU to load half of.
. "$(dirname "$0")/../_lib/preamble.sh"

XDECK=$(upstream xfluid_ls_neumann_inflow_stab.4C.yaml)  || exit 3
PDECK=$(upstream f3_stokes_residualbased_rotboxgeom.4C.yaml) || exit 3
grep -q "^FLUID DYNAMIC:" "$XDECK" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "^FLUID DYNAMIC:" "$PDECK" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

VTK='IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/FLUID:
  OUTPUT_FLUID: true
  VELOCITY: true
  PRESSURE: true
FLUID DYNAMIC:
'
add_vtk() { python3 -c "
import sys
t=open(sys.argv[1]).read()
open(sys.argv[2],'w').write(t.replace('FLUID DYNAMIC:\n', sys.argv[3], 1))" "$1" "$2" "$VTK"; }

add_vtk "$PDECK" "$TMP/plain.yaml"
add_vtk "$XDECK" "$TMP/xfem.yaml"

mkdir -p "$TMP/pout" "$TMP/xout"
run4c "$TMP/plain.yaml" "$TMP/pout/o" > "$TMP/PLAIN.log" 2>&1; echo "EXIT_PLAIN=$?"
run4c "$TMP/xfem.yaml"  "$TMP/xout/o" > "$TMP/XFEM.log"  2>&1; echo "EXIT_XFEM=$?"

echo "PLAIN_VTU=$(find "$TMP/pout" -name '*.vtu' | wc -l)"
echo "PLAIN_PVD=$(find "$TMP/pout" -name '*.pvd' | wc -l)"
echo "XFEM_VTU=$(find "$TMP/xout" -name '*.vtu' | wc -l)"
echo "XFEM_PVD=$(find "$TMP/xout" -name '*.pvd' | wc -l)"
echo "XFEM_SUBDOMAIN_VTU=$(find "$TMP/xout" -name '*subdomain*' | wc -l)"
# The XFEM run is otherwise perfectly healthy and silent about the missing output.
grep -m1 -F "processor 0 finished normally" "$TMP/XFEM.log"
echo "XFEM_OUTPUT_WARNINGS=$(grep -ciE 'vtu|vtk output.*(not|unsupported)' "$TMP/XFEM.log")"
find "$TMP/pout" -name '*.vtu' -printf '%f\n' | head -1
exit 0
