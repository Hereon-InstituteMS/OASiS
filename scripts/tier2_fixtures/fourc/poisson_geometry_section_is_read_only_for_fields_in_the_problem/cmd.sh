#!/bin/bash

# Tier-2 for fourc::poisson#3 — which geometry sections are read, which are
# ignored, and which are refused by name.  The entry says a misdirected
# geometry section "gives no message at all"; that is true only when the field
# it names is not part of this problem type, and this fixture pins both halves.
#
#   INLINE     no geometry section, no external file  -> exit 0
#   STRUCTURE  'STRUCTURE GEOMETRY' naming a file that does not exist -> exit
#              0.  A Scalar_Transport problem has no structure field, so the
#              section is syntactically valid and never read.
#   FLUID      'FLUID GEOMETRY' naming the same missing file -> exit 1.  A
#              Scalar_Transport problem DOES carry a fluid discretization, so
#              this route is taken and the missing file is reported by name.
#   TRANSPORT  'TRANSPORT GEOMETRY' alongside the inline sections -> exit 1,
#              both competing routes listed
#   POISSONG   'POISSON GEOMETRY' -> not a field at all, rejected by name
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

BLOCK_TRANSP='  FILE: \"absent_grid.exo\"\n  ELEMENT_BLOCKS:\n    - ID: 1\n      TRANSP:\n        TRI3:\n          MAT: 1\n          TYPE: Std\n'
BLOCK_SOLID='  FILE: \"absent_grid.exo\"\n  ELEMENT_BLOCKS:\n    - ID: 1\n      SOLID:\n        HEX8:\n          MAT: 1\n          KINEM: nonlinear\n'
BLOCK_FLUID='  FILE: \"absent_grid.exo\"\n  ELEMENT_BLOCKS:\n    - ID: 1\n      FLUID:\n        TRI3:\n          MAT: 1\n          NA: Euler\n'

pois "{}"                                                        "$TMP/inline.yaml"
pois "{\"extra_sections\":\"STRUCTURE GEOMETRY:\n$BLOCK_SOLID\"}"   "$TMP/structure.yaml"
pois "{\"extra_sections\":\"FLUID GEOMETRY:\n$BLOCK_FLUID\"}"       "$TMP/fluid.yaml"
pois "{\"extra_sections\":\"TRANSPORT GEOMETRY:\n$BLOCK_TRANSP\"}"  "$TMP/transport.yaml"
pois "{\"extra_sections\":\"POISSON GEOMETRY:\n$BLOCK_TRANSP\"}"    "$TMP/poissong.yaml"

probe INLINE    "$TMP/inline.yaml"
probe STRUCTURE "$TMP/structure.yaml"
probe FLUID     "$TMP/fluid.yaml"
probe TRANSPORT "$TMP/transport.yaml"
probe POISSONG  "$TMP/poissong.yaml"

# 1. The inline route needs no external file.
grep -m1 -F "processor 0 finished normally" "$TMP/INLINE.log"

# 2. A field this problem does not have: valid name, never read, silent.
grep -m1 -F "processor 0 finished normally" "$TMP/STRUCTURE.log"
echo "STRUCTURE_GEOM_TOUCHED_THE_FILE=$(grep -ci 'absent_grid.exo' "$TMP/STRUCTURE.log")"
echo "STRUCTURE_GEOM_SECTION_NAME_REJECTED=$(grep -c 'not a valid section name' "$TMP/STRUCTURE.log")"

# 3. A field this problem DOES have: the route is taken, the read is announced
#    and the missing file is named.  4C resolves FILE relative to the deck's
#    own directory, so only the tail of the path is stable across runs.
echo "FLUID_GEOM_ANNOUNCED_THE_READ=$(grep -c 'Read mesh from file' "$TMP/FLUID.log")"
grep -m1 -oF "absent_grid.exo does not exist." "$TMP/FLUID.log"
grep -m1 -F "4C_io_exodus.cpp" "$TMP/FLUID.log"

# 4. Two routes for the scatra mesh: refused, with both names listed.
grep -m1 -F "Multiple options to read mesh for discretization 'scatra'. Only one is allowed." "$TMP/TRANSPORT.log"
grep -m1 -F "'TRANSPORT ELEMENTS' 'TRANSPORT GEOMETRY'" "$TMP/TRANSPORT.log"
grep -m1 -F "4C_io_meshreader.cpp" "$TMP/TRANSPORT.log"

# 5. Not a field at all: rejected by name before anything runs.
grep -m1 -F "Section 'POISSON GEOMETRY' is not a valid section name." "$TMP/POISSONG.log"

# The string an earlier version of this entry quoted is in none of the logs.
echo "CLAIMED_ELEMENT_CATEGORY_TEXT=$(cat "$TMP"/INLINE.log "$TMP"/STRUCTURE.log "$TMP"/FLUID.log \
  "$TMP"/TRANSPORT.log "$TMP"/POISSONG.log | grep -ci 'expected element category')"
exit 0
