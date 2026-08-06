#!/bin/bash

# Tier-2 for fourc::fluid#0 — the fluid section names, and the one wrong name
# that section-name validation does NOT catch.
#
# On a self-contained 2-D deck (unit square of FLUID QUAD4, MAT_fluid, u = 1
# driven in on the left edge, no-slip top and bottom, pressure pinned at a
# corner):
#
#   BASE       'FLUID DYNAMIC' + 'FLUID ELEMENTS' -> exit 0, result test CORRECT
#   FLUIDSEC   'FLUID'              -> rejected by name
#   FLUIDDYN   'FLUID_DYN'          -> rejected by name
#   STRUCTSEC  'STRUCTURE'          -> rejected by name, so the bare word the
#                                      entry warns about is not even a section
#   STRUCTELE  'STRUCTURE ELEMENTS' -> ACCEPTED by name.  It is a real section,
#                                      just not the fluid one, so the elements
#                                      go nowhere, the fluid discretization is
#                                      left empty, and the abort blames the
#                                      wrong thing: "Pressure map empty. Wrong
#                                      DIM value in input file?"
#
# The last arm is the one worth knowing: the diagnostic sends you to PROBLEM
# SIZE/DIM, which is correct in this deck.
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

# One generator emits the whole 2-D fluid deck, so the arms differ ONLY in the
# knob under test: a unit square of FLUID QUAD4 with MAT_fluid, u = 1 driven
# in on the left edge, no-slip top and bottom, pressure pinned at one corner.
fluid() {  # $1 = JSON knobs, $2 = output file
python3 - "$1" > "$2" <<'FLUIDPY'
import sys, json
k = json.loads(sys.argv[1])
n        = k.get("n", 6)
dynsect  = k.get("dyn_section", "FLUID DYNAMIC")
elesect  = k.get("element_section", "FLUID ELEMENTS")
visc     = k.get("visc", 0.01)
timeint  = k.get("timeintegr", "One_Step_Theta")
dt       = k.get("dt", 0.1)
numstep  = k.get("numstep", 5)
stab     = k.get("stab_section", "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION")
stabbody = k.get("stab_body", "")
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
        els.append(f"{len(els)+1} FLUID QUAD4 {ids[(i,j)]} {ids[(i+1,j)]} "
                   f"{ids[(i+1,j+1)]} {ids[(i,j+1)]} MAT 1 NA Euler")
left   = [ids[(0, j)] for j in range(1, n)]
top    = [ids[(i, n)] for i in range(n + 1)]
bottom = [ids[(i, 0)] for i in range(n + 1)]
corner = ids[(n, 0)]
stabblk = (stab + ":\n" + stabbody) if stabbody else ""
y = f'''PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Fluid"
{dynsect}:
  LINEAR_SOLVER: 1
  TIMEINTEGR: "{timeint}"
  THETA: 1.0
  TIMESTEP: {dt:.16g}
  NUMSTEP: {numstep}
  MAXTIME: {dt*numstep:.16g}
  ITEMAX: 20
  INITIALFIELD: "zero_field"
{stabblk}SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Fluid_Solver"
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: {visc:.16g}
      DENSITY: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1.0, 0.0, 0.0]
    FUNCT: [1, 0, 0]
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 3
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
{extra}'''
if results:
    y += "RESULT DESCRIPTION:\n"
    for node, q, v, tol in results:
        y += (f'  - FLUID:\n      DIS: "fluid"\n      NODE: {node}\n'
              f'      QUANTITY: "{q}"\n      VALUE: {v:.17g}\n'
              f'      TOLERANCE: {tol:.17g}\n')
y += "DNODE-NODE TOPOLOGY:\n" + f'  - "NODE {corner} DNODE 1"\n'
y += "DLINE-NODE TOPOLOGY:\n"
y += "".join(f'  - "NODE {i} DLINE 1"\n' for i in left)
y += "".join(f'  - "NODE {i} DLINE 2"\n' for i in top)
y += "".join(f'  - "NODE {i} DLINE 3"\n' for i in bottom)
y += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in coords)
y += elesect + ":\n" + "".join(f'  - "{s}"\n' for s in els)
sys.stdout.write(y)
FLUIDPY
}

NODE=25
TRUE=1.01714304698809932

fluid "{\"results\":[[$NODE,\"velx\",$TRUE,1e-10]]}" "$TMP/base.yaml"
fluid "{\"dyn_section\":\"FLUID\"}"                   "$TMP/fluidsec.yaml"
fluid "{\"dyn_section\":\"FLUID_DYN\"}"               "$TMP/fluiddyn.yaml"
fluid "{\"element_section\":\"STRUCTURE\"}"           "$TMP/structsec.yaml"
fluid "{\"element_section\":\"STRUCTURE ELEMENTS\"}"  "$TMP/structele.yaml"

probe BASE      "$TMP/base.yaml"
probe FLUIDSEC  "$TMP/fluidsec.yaml"
probe FLUIDDYN  "$TMP/fluiddyn.yaml"
probe STRUCTSEC "$TMP/structsec.yaml"
probe STRUCTELE "$TMP/structele.yaml"

# The right names solve the problem.
grep -m1 -F "is CORRECT" "$TMP/BASE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"

# Three wrong names are caught by section-name validation, verbatim.
grep -m1 -F "Section 'FLUID' is not a valid section name." "$TMP/FLUIDSEC.log"
grep -m1 -F "Section 'FLUID_DYN' is not a valid section name." "$TMP/FLUIDDYN.log"
grep -m1 -F "Section 'STRUCTURE' is not a valid section name." "$TMP/STRUCTSEC.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/FLUIDSEC.log"

# The fourth is not caught by name at all, and the abort names DIM instead.
echo "STRUCTELE_SECTION_NAME_REJECTED=$(grep -c 'not a valid section name' "$TMP/STRUCTELE.log")"
grep -m1 -F "Pressure map empty. Wrong DIM value in input file?" "$TMP/STRUCTELE.log"
grep -m1 -F "4C_fluid_implicit_integration.cpp" "$TMP/STRUCTELE.log"
echo "STRUCTELE_BLAMED_THE_ELEMENT_SECTION=$(grep -ci 'FLUID ELEMENTS' "$TMP/STRUCTELE.log")"
exit 0
