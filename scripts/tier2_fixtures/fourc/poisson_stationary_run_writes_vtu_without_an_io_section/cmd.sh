#!/bin/bash

# Tier-2 for fourc::poisson#2 — even a one-step stationary solve writes its
# VTK without being asked, and the output sections an agent invents for scalar
# transport are fatal.
#
#   AUTO         no output section at all -> exit 0, and the run writes
#                <prefix>-vtk-files/scatra-*.vtu (initial state and solution)
#                plus <prefix>-scatra.pvd, the nodal array named phi_1
#   SCATRA_SECT  'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' -> exit 1
#   IO_SECT      'IO/RUNTIME VTK OUTPUT/SCATRA'                -> exit 1
#   IO_FLUID     'IO/RUNTIME VTK OUTPUT/FLUID'                 -> valid name,
#                exit 0 — so the two rejections above are about those names,
#                not about output sections being unwelcome here
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

# One generator emits the whole stationary Poisson deck, so the arms differ
# ONLY in the knob under test: a unit square of TRANSP TRI3 with MAT_scatra,
# TIMEINTEGR "Stationary", phi = 1 on the left edge and the right edge either
# held at 0 (Dirichlet) or given a flux (Neumann).
pois() {  # $1 = JSON knobs, $2 = output file
python3 - "$1" > "$2" <<'POISPY'
import sys, json
k = json.loads(sys.argv[1])
n        = k.get("n", 4)
dynsect  = k.get("dyn_section", "SCALAR TRANSPORT DYNAMIC")
timeint  = k.get("timeintegr", "Stationary")
theta    = k.get("theta", 1.0)
dt       = k.get("dt", 1.0)
numstep  = k.get("numstep", 1)
velfield = k.get("velocityfield", '  VELOCITYFIELD: "zero"\n')
numdof   = k.get("numdof", 1)
onoff    = k.get("onoff", "[1]")
val      = k.get("val", "[1.0]")
funct    = k.get("funct", "[1]")
right    = k.get("right_condition", "dirich")
rval     = k.get("right_val", "")
extra    = k.get("extra_sections", "")
results  = k.get("results", [])
ids, coords, nid = {}, [], 0
for j in range(n + 1):
    for i in range(n + 1):
        nid += 1
        ids[(i, j)] = nid
        coords.append(f"NODE {nid} COORD {i/n:.16g} {j/n:.16g} 0.0")
els = []
for j in range(n):
    for i in range(n):
        a, b, c, d = ids[(i, j)], ids[(i+1, j)], ids[(i+1, j+1)], ids[(i, j+1)]
        els.append(f"{len(els)+1} TRANSP TRI3 {a} {b} {c} MAT 1 TYPE Std")
        els.append(f"{len(els)+1} TRANSP TRI3 {a} {c} {d} MAT 1 TYPE Std")
lefte  = [ids[(0, j)] for j in range(n + 1)]
righte = [ids[(n, j)] for j in range(n + 1)]
zeros = "[0.0]" if numdof == 1 else "[" + ", ".join(["0.0"] * numdof) + "]"
zfun = "[0]" if numdof == 1 else "[" + ", ".join(["0"] * numdof) + "]"
if not rval:
    rval = zeros
cond = f'''DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: {numdof}
    ONOFF: {onoff}
    VAL: {val}
    FUNCT: {funct}
'''
if right == "dirich":
    cond += f'''  - E: 2
    NUMDOF: {numdof}
    ONOFF: {onoff}
    VAL: {rval}
    FUNCT: {zfun}
'''
elif right == "neumann":
    cond += f'''DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: {numdof}
    ONOFF: {onoff}
    VAL: {rval}
    FUNCT: {zfun}
'''
y = f'''PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
{dynsect}:
  SOLVERTYPE: "linear_full"
  TIMEINTEGR: "{timeint}"
  THETA: {theta:.16g}
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
{cond}{extra}'''
if results:
    y += "RESULT DESCRIPTION:\n"
    for node, v, tol in results:
        y += (f'  - SCATRA:\n      DIS: "scatra"\n      NODE: {node}\n'
              f'      QUANTITY: "phi"\n      VALUE: {v:.17g}\n'
              f'      TOLERANCE: {tol:.17g}\n')
y += "DLINE-NODE TOPOLOGY:\n"
y += "".join(f'  - "NODE {i} DLINE 1"\n' for i in lefte)
y += "".join(f'  - "NODE {i} DLINE 2"\n' for i in righte)
y += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in coords)
y += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{s}"\n' for s in els)
sys.stdout.write(y)
POISPY
}

pois "{}" "$TMP/auto.yaml"
pois "{\"extra_sections\":\"SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT:\n  OUTPUT_SCATRA: true\n\"}" \
     "$TMP/scatra_sect.yaml"
pois "{\"extra_sections\":\"IO/RUNTIME VTK OUTPUT/SCATRA:\n  OUTPUT_SCATRA: true\n\"}" \
     "$TMP/io_sect.yaml"
pois "{\"extra_sections\":\"IO/RUNTIME VTK OUTPUT/FLUID:\n  OUTPUT_FLUID: true\n\"}" \
     "$TMP/io_fluid.yaml"

echo "AUTO_DECK_HAS_OUTPUT_SECTION=$(grep -cE '^(IO|SCALAR TRANSPORT DYNAMIC)/' "$TMP/auto.yaml")"

probe AUTO        "$TMP/auto.yaml"
probe SCATRA_SECT "$TMP/scatra_sect.yaml"
probe IO_SECT     "$TMP/io_sect.yaml"
probe IO_FLUID    "$TMP/io_fluid.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/AUTO.log"
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

grep -m1 -F "Section 'SCALAR TRANSPORT DYNAMIC/RUNTIME VTK OUTPUT' is not a valid section name." "$TMP/SCATRA_SECT.log"
grep -m1 -F "Section 'IO/RUNTIME VTK OUTPUT/SCATRA' is not a valid section name." "$TMP/IO_SECT.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/SCATRA_SECT.log"

# The discriminating arm: a runtime-VTK subsection for a field that DOES exist
# in the schema is accepted, so the rejections above are name-specific.
grep -m1 -F "processor 0 finished normally" "$TMP/IO_FLUID.log"
echo "IO_FLUID_SECTION_NAME_REJECTED=$(grep -c 'not a valid section name' "$TMP/IO_FLUID.log")"
exit 0
