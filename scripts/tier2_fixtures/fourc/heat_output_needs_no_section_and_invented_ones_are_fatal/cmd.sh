#!/bin/bash

# Tier-2 for fourc::heat#2 — a Scalar_Transport run writes its VTK without
# being asked, and the two output sections an agent is most likely to invent
# are fatal rather than merely ineffective.
#
#   AUTO         no output section of any kind -> exit 0, and the run still
#                writes <prefix>-vtk-files/scatra-*.vtu and
#                <prefix>-scatra.pvd, the nodal array named phi_1
#   IOVTK        'IO/RUNTIME VTK OUTPUT'                       -> valid name,
#                exit 0, and it produces no scatra file of its own
#   SCATRA_SECT  'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' -> exit 1
#   IO_SECT      'IO/RUNTIME VTK OUTPUT/SCATRA'                -> exit 1
#
# The IOVTK arm is the discriminating one: it shows the two rejections are
# about those particular names and not about output sections in general.
# --- self-contained preamble (deliberately NOT sourced from ../_lib) --------
# scripts/mutate_tier2_fixtures.py stages this directory into a scratch tree.
# Everything the fixture needs is inline, so the mutation proof cannot be
# confounded by a missing sibling.  Same honesty rule as the shared preamble:
# when 4C is missing this prints FIXTURE_ABORT=no_binary and exits non-zero,
# and fixture.json forbids both strings, so an absent solver makes the fixture
# RED rather than green.
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" \
          "$HOME/Schreibtisch/4C-src/4C/build/4C" "/usr/local/bin/4C"; do
  [ -n "${_c:-}" ] && [ -x "$_c" ] && BIN="$_c" && break
done
if [ -z "${BIN:-}" ]; then
  echo "FIXTURE_ABORT=no_binary (set FOURC_BINARY to a 4C executable)"
  exit 3
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# stdbuf is not decoration: 4C writes result-test verdicts to raw std::cout
# and MPI_Abort discards a block-buffered stdout (pitfall input_format#18).
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }
probe() { run4c "$2" "$TMP/o_$1" > "$TMP/$1.log" 2>&1; echo "EXIT_$1=$?"; }
# ---------------------------------------------------------------------------

# One generator emits the whole 3-D heat-conduction deck, so the arms differ
# ONLY in the knob under test: a unit cube of TRANSP HEX8 with MAT_scatra,
# phi = 1 held on the x=0 face and 0 on the x=1 face.
heat() {  # $1 = JSON knobs, $2 = output file
python3 - "$1" > "$2" <<'HEATPY'
import sys, json
k = json.loads(sys.argv[1])
n        = k.get("n", 3)
dynsect  = k.get("dyn_section", "SCALAR TRANSPORT DYNAMIC")
timeint  = k.get("timeintegr", "One_Step_Theta")
dt       = k.get("dt", 0.02)
numstep  = k.get("numstep", 10)
velfield = k.get("velocityfield", '  VELOCITYFIELD: "zero"\n')
numdof   = k.get("numdof", 1)
onoff    = k.get("onoff", "[1]")
val      = k.get("val", "[1.0]")
funct    = k.get("funct", "[1]")
extra    = k.get("extra_sections", "")
results  = k.get("results", [])
ids, coords, nid = {}, [], 0
for kk in range(n + 1):
    for j in range(n + 1):
        for i in range(n + 1):
            nid += 1
            ids[(i, j, kk)] = nid
            coords.append(f"NODE {nid} COORD {i/n:.16g} {j/n:.16g} {kk/n:.16g}")
els = []
for kk in range(n):
    for j in range(n):
        for i in range(n):
            c = [ids[(i, j, kk)], ids[(i+1, j, kk)], ids[(i+1, j+1, kk)],
                 ids[(i, j+1, kk)], ids[(i, j, kk+1)], ids[(i+1, j, kk+1)],
                 ids[(i+1, j+1, kk+1)], ids[(i, j+1, kk+1)]]
            els.append(f"{len(els)+1} TRANSP HEX8 " + " ".join(str(x) for x in c)
                       + " MAT 1 TYPE Std")
hot  = [ids[(0, j, kk)] for kk in range(n + 1) for j in range(n + 1)]
cold = [ids[(n, j, kk)] for kk in range(n + 1) for j in range(n + 1)]
zeros = "[0.0]" if numdof == 1 else "[" + ", ".join(["0.0"] * numdof) + "]"
zfun = "[0]" if numdof == 1 else "[" + ", ".join(["0"] * numdof) + "]"
y = f'''PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
{dynsect}:
  SOLVERTYPE: "linear_full"
  TIMEINTEGR: "{timeint}"
  THETA: 1.0
  TIMESTEP: {dt:.16g}
  NUMSTEP: {numstep}
  MAXTIME: {dt*numstep:.16g}
{velfield}  INITIALFIELD: "zero_field"
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Scatra_Solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: {numdof}
    ONOFF: {onoff}
    VAL: {val}
    FUNCT: {funct}
  - E: 2
    NUMDOF: {numdof}
    ONOFF: {onoff}
    VAL: {zeros}
    FUNCT: {zfun}
{extra}'''
if results:
    y += "RESULT DESCRIPTION:\n"
    for node, v, tol in results:
        y += (f'  - SCATRA:\n      DIS: "scatra"\n      NODE: {node}\n'
              f'      QUANTITY: "phi"\n      VALUE: {v:.17g}\n'
              f'      TOLERANCE: {tol:.17g}\n')
y += "DSURF-NODE TOPOLOGY:\n"
y += "".join(f'  - "NODE {i} DSURFACE 1"\n' for i in hot)
y += "".join(f'  - "NODE {i} DSURFACE 2"\n' for i in cold)
y += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in coords)
y += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{s}"\n' for s in els)
sys.stdout.write(y)
HEATPY
}

heat "{\"numstep\":4}" "$TMP/auto.yaml"
heat "{\"numstep\":2,\"extra_sections\":\"IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n\"}" \
     "$TMP/iovtk.yaml"
heat "{\"numstep\":2,\"extra_sections\":\"SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT:\n  OUTPUT_SCATRA: true\n\"}" \
     "$TMP/scatra_sect.yaml"
heat "{\"numstep\":2,\"extra_sections\":\"IO/RUNTIME VTK OUTPUT/SCATRA:\n  OUTPUT_SCATRA: true\n\"}" \
     "$TMP/io_sect.yaml"

# The baseline really does ask for nothing.
echo "AUTO_DECK_HAS_OUTPUT_SECTION=$(grep -cE '^(IO|SCALAR TRANSPORT DYNAMIC)/' "$TMP/auto.yaml")"

probe AUTO        "$TMP/auto.yaml"
probe IOVTK       "$TMP/iovtk.yaml"
probe SCATRA_SECT "$TMP/scatra_sect.yaml"
probe IO_SECT     "$TMP/io_sect.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/AUTO.log"

# ...and wrote the VTK anyway.
if [ -d "$TMP/o_AUTO-vtk-files" ]; then
  echo "AUTO_VTU_COUNT=$(ls "$TMP/o_AUTO-vtk-files"/scatra-*.vtu 2>/dev/null | wc -l)"
else
  echo "AUTO_VTU_COUNT=0"
fi
if [ -f "$TMP/o_AUTO-scatra.pvd" ]; then echo "AUTO_PVD=yes"; else echo "AUTO_PVD=no"; fi
if grep -qh 'Name="phi_1"' "$TMP"/o_AUTO-vtk-files/scatra-*.vtu 2>/dev/null; then
  echo "AUTO_VTU_ARRAY_NAME=phi_1"
else
  echo "AUTO_VTU_ARRAY_NAME=absent"
fi

# A real output section is accepted by name and adds no scatra file.
grep -m1 -F "processor 0 finished normally" "$TMP/IOVTK.log"
echo "IOVTK_SECTION_NAME_REJECTED=$(grep -c 'not a valid section name' "$TMP/IOVTK.log")"

# The two invented names never get as far as running.
grep -m1 -F "Section 'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' is not a valid section name." "$TMP/SCATRA_SECT.log"
grep -m1 -F "Section 'IO/RUNTIME VTK OUTPUT/SCATRA' is not a valid section name." "$TMP/IO_SECT.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/IO_SECT.log"
exit 0
