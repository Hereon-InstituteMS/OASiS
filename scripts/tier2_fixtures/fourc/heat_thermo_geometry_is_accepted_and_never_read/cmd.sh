#!/bin/bash

# Tier-2 for fourc::heat#3 — the scatra mesh routes, and the trap that a
# geometry section naming the WRONG field is not caught by anything.
#
#   INLINE   NODE COORDS + TRANSPORT ELEMENTS + D*-NODE TOPOLOGY, no geometry
#            section and no external file    -> exit 0
#   THERMO   'THERMO GEOMETRY' pointing at a file that does not exist -> exit
#            0.  A heat problem in 4C is a scatra problem, THERMO GEOMETRY is
#            a perfectly valid section name, and it is simply never read: the
#            .exo it names is never touched and no message is printed.
#   TRANSP   'TRANSPORT GEOMETRY' *is* the scatra route, so having it as well
#            as the inline sections is refused, with both names listed
#   HEATGEOM 'HEAT GEOMETRY' is not a field, so the name itself is rejected
#
# The string an earlier version of this entry quoted, 'expected element
# category TRANSP but got SOLID', appears in none of the four logs.
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

BLOCK_TRANSP='  FILE: \"missing_mesh.exo\"\n  ELEMENT_BLOCKS:\n    - ID: 1\n      TRANSP:\n        HEX8:\n          MAT: 1\n          TYPE: Std\n'

heat "{\"numstep\":2}"                                                    "$TMP/inline.yaml"
heat "{\"numstep\":2,\"extra_sections\":\"THERMO GEOMETRY:\n$BLOCK_TRANSP\"}"    "$TMP/thermo.yaml"
heat "{\"numstep\":2,\"extra_sections\":\"TRANSPORT GEOMETRY:\n$BLOCK_TRANSP\"}" "$TMP/transp.yaml"
heat "{\"numstep\":2,\"extra_sections\":\"HEAT GEOMETRY:\n$BLOCK_TRANSP\"}"      "$TMP/heatgeom.yaml"

probe INLINE   "$TMP/inline.yaml"
probe THERMO   "$TMP/thermo.yaml"
probe TRANSP   "$TMP/transp.yaml"
probe HEATGEOM "$TMP/heatgeom.yaml"

# 1. The inline route needs no external file at all.
grep -m1 -F "processor 0 finished normally" "$TMP/INLINE.log"

# 2. THERMO GEOMETRY is a valid name and is never read for the scatra field:
#    the file it points at does not exist and the run finishes regardless.
grep -m1 -F "processor 0 finished normally" "$TMP/THERMO.log"
echo "THERMO_GEOM_TOUCHED_THE_FILE=$(grep -ci 'missing_mesh.exo' "$TMP/THERMO.log")"
echo "THERMO_GEOM_SECTION_NAME_REJECTED=$(grep -c 'not a valid section name' "$TMP/THERMO.log")"

# 3. TRANSPORT GEOMETRY is the scatra route, and two routes at once are
#    refused with both section names spelled out.
grep -m1 -F "Multiple options to read mesh for discretization 'scatra'. Only one is allowed." "$TMP/TRANSP.log"
grep -m1 -F "'TRANSPORT ELEMENTS' 'TRANSPORT GEOMETRY'" "$TMP/TRANSP.log"
grep -m1 -F "4C_io_meshreader.cpp" "$TMP/TRANSP.log"

# 4. A field name that does not exist is rejected before anything runs.
grep -m1 -F "Section 'HEAT GEOMETRY' is not a valid section name." "$TMP/HEATGEOM.log"

# 5. The string the entry used to quote is in none of the logs.
echo "CLAIMED_ELEMENT_CATEGORY_TEXT=$(cat "$TMP"/INLINE.log "$TMP"/THERMO.log \
  "$TMP"/TRANSP.log "$TMP"/HEATGEOM.log | grep -ci 'expected element category')"
exit 0
