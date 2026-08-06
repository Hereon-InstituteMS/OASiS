#!/bin/bash

# Tier-2 for fourc::scalar_transport#2 — scalar transport has NO runtime-VTK
# section, needs none, and naming one is fatal rather than merely useless.
#
#   AUTO            deck with no output section whatsoever -> exit 0, and the
#                   run writes <prefix>-vtk-files/scatra-*.vtu plus
#                   <prefix>-scatra.pvd anyway, the nodal array named phi_1
#   SCATRA_SECTION  'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' -> exit 1
#   IO_SECTION      'IO/RUNTIME VTK OUTPUT/SCATRA'                -> exit 1
#
# Both rejections come from section-name validation in
# core/io/src/4C_io_input_file.cpp, before anything runs.
# --- self-contained preamble (deliberately NOT sourced from ../_lib) --------
# scripts/mutate_tier2_fixtures.py copies ONLY this directory into a scratch
# tree.  A fixture that sources ../_lib/preamble.sh therefore cannot even
# start there, its mutant dies for the wrong reason, and the KILLED verdict
# certifies nothing.  Everything this fixture needs is inline, so the
# mutation proof is real.  Same honesty rule as the shared preamble: when 4C
# is missing this prints FIXTURE_ABORT=no_binary and exits non-zero, and
# fixture.json forbids both strings, so an absent solver makes the fixture
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

# One small generator emits the whole scalar-transport deck, so the arms
# differ ONLY in the knob under test.  Pure diffusion on a unit square,
# phi = 1 on the left edge and 0 on the right, TRANSP QUAD4 + MAT_scatra.
sca() {  # $1 = JSON knobs, $2 = output file
python3 - "$1" > "$2" <<'SCAPY'
import sys, json
k = json.loads(sys.argv[1])
n         = k.get("n", 4)
timeint   = k.get("timeintegr", "One_Step_Theta")
dt        = k.get("dt", 0.05)
numstep   = k.get("numstep", 20)
velfield  = k.get("velocityfield", '  VELOCITYFIELD: "zero"\n')
numdof    = k.get("numdof", 1)
onoff     = k.get("onoff", "[1]")
val       = k.get("val", "[1.0]")
funct     = k.get("funct", "[1]")
elesect   = k.get("element_section", "TRANSPORT ELEMENTS")
extra     = k.get("extra_sections", "")
results   = k.get("results", [])
ids, coords, nid = {}, [], 0
for j in range(n + 1):
    for i in range(n + 1):
        nid += 1
        ids[(i, j)] = nid
        coords.append(f"NODE {nid} COORD {i/n:.16g} {j/n:.16g} 0.0")
els = []
for j in range(n):
    for i in range(n):
        els.append(f"{len(els)+1} TRANSP QUAD4 {ids[(i,j)]} {ids[(i+1,j)]} "
                   f"{ids[(i+1,j+1)]} {ids[(i,j+1)]} MAT 1 TYPE Std")
left  = [ids[(0, j)] for j in range(n + 1)]
right = [ids[(n, j)] for j in range(n + 1)]
zeros = ("[0.0]" if numdof == 1 else "[" + ", ".join(["0.0"]*numdof) + "]")
zfun  = ("[0]"   if numdof == 1 else "[" + ", ".join(["0"]*numdof) + "]")
y = f'''PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
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
DESIGN LINE DIRICH CONDITIONS:
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
y += "DLINE-NODE TOPOLOGY:\n"
y += "".join(f'  - "NODE {i} DLINE 1"\n' for i in left)
y += "".join(f'  - "NODE {i} DLINE 2"\n' for i in right)
y += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in coords)
y += elesect + ":\n" + "".join(f'  - "{s}"\n' for s in els)
sys.stdout.write(y)
SCAPY
}

sca "{\"numstep\":4,\"dt\":0.1}" "$TMP/auto.yaml"
sca "{\"numstep\":2,\"dt\":0.1,\"extra_sections\":\"SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT:\n  OUTPUT_SCATRA: true\n\"}" \
    "$TMP/scatra_section.yaml"
sca "{\"numstep\":2,\"dt\":0.1,\"extra_sections\":\"IO/RUNTIME VTK OUTPUT/SCATRA:\n  OUTPUT_SCATRA: true\n\"}" \
    "$TMP/io_section.yaml"

# The baseline really does carry no output section of any kind.
echo "AUTO_DECK_HAS_IO_SECTION=$(grep -cE '^(IO|SCALAR TRANSPORT DYNAMIC)/' "$TMP/auto.yaml")"

probe AUTO           "$TMP/auto.yaml"
probe SCATRA_SECTION "$TMP/scatra_section.yaml"
probe IO_SECTION     "$TMP/io_section.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/AUTO.log"
grep -m1 -F "Section 'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' is not a valid section name." "$TMP/SCATRA_SECTION.log"
grep -m1 -F "Section 'IO/RUNTIME VTK OUTPUT/SCATRA' is not a valid section name." "$TMP/IO_SECTION.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/SCATRA_SECTION.log"

# ...and the run that asked for nothing wrote the VTK anyway.
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
exit 0
