#!/bin/bash

# Tier-2 for fourc::scalar_transport#3, and the execution that CORRECTED it.
#
# Three things, one deck each:
#
#  1. The inline route (NODE COORDS + TRANSPORT ELEMENTS + D*-NODE TOPOLOGY)
#     needs no geometry section and no external file at all.
#  2. 'STRUCTURE GEOMETRY' in a Scalar_Transport deck is NOT caught by section
#     -name validation — every known field has a GEOMETRY section, so the name
#     is valid.  It is simply never read for the scatra field: the run exits 0
#     although the file it names does not exist.  An agent who writes it
#     instead of TRANSPORT GEOMETRY gets no error at all.
#  3. 'TRANSPORT GEOMETRY' IS the scatra route, and when it is present
#     alongside the inline section 4C enumerates the competing routes by name.
#
# The entry's Signal — a section name that does not exist being rejected with
# "is not a valid section name." — is real, and the fourth arm shows it with
# 'SCATRA GEOMETRY'.  The string an earlier version quoted, 'expected element
# category TRANSP but got SOLID', appears in none of the four logs.
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

BLOCK_SOLID='  FILE: \"nonexistent.exo\"\n  ELEMENT_BLOCKS:\n    - ID: 1\n      SOLID:\n        HEX8:\n          MAT: 1\n          KINEM: nonlinear\n'
BLOCK_TRANSP='  FILE: \"nonexistent.exo\"\n  ELEMENT_BLOCKS:\n    - ID: 1\n      TRANSP:\n        QUAD4:\n          MAT: 1\n          TYPE: Std\n'

sca "{\"numstep\":2,\"dt\":0.1}" "$TMP/inline.yaml"
sca "{\"numstep\":2,\"dt\":0.1,\"extra_sections\":\"STRUCTURE GEOMETRY:\n$BLOCK_SOLID\"}"   "$TMP/structure_geom.yaml"
sca "{\"numstep\":2,\"dt\":0.1,\"extra_sections\":\"TRANSPORT GEOMETRY:\n$BLOCK_TRANSP\"}" "$TMP/transport_geom.yaml"
sca "{\"numstep\":2,\"dt\":0.1,\"extra_sections\":\"SCATRA GEOMETRY:\n$BLOCK_TRANSP\"}"    "$TMP/scatra_geom.yaml"

probe INLINE          "$TMP/inline.yaml"
probe STRUCTURE_GEOM  "$TMP/structure_geom.yaml"
probe TRANSPORT_GEOM  "$TMP/transport_geom.yaml"
probe SCATRA_GEOM     "$TMP/scatra_geom.yaml"

# 1. The inline route needs no external file at all.
grep -m1 -F "processor 0 finished normally" "$TMP/INLINE.log"

# 2. STRUCTURE GEOMETRY is a VALID section name, so it is not rejected — it is
#    simply never read for the scatra field.  The file it names does not exist
#    and the run still finishes.
grep -m1 -F "processor 0 finished normally" "$TMP/STRUCTURE_GEOM.log"
echo "STRUCTURE_GEOM_TOUCHED_THE_FILE=$(grep -ci 'nonexistent.exo' "$TMP/STRUCTURE_GEOM.log")"
echo "STRUCTURE_GEOM_SECTION_NAME_REJECTED=$(grep -c 'not a valid section name' "$TMP/STRUCTURE_GEOM.log")"

# 3. TRANSPORT GEOMETRY *is* the scatra route, and 4C enumerates the competing
#    routes by name when more than one is present.
grep -m1 -F "Multiple options to read mesh for discretization 'scatra'. Only one is allowed." "$TMP/TRANSPORT_GEOM.log"
grep -m1 -F "'TRANSPORT ELEMENTS' 'TRANSPORT GEOMETRY'" "$TMP/TRANSPORT_GEOM.log"
grep -m1 -F "4C_io_meshreader.cpp" "$TMP/TRANSPORT_GEOM.log"

# 4. A field name that does not exist IS rejected by name — the entry's Signal.
grep -m1 -F "Section 'SCATRA GEOMETRY' is not a valid section name." "$TMP/SCATRA_GEOM.log"

# 5. The string the entry used to quote is in none of the four logs.
echo "CLAIMED_ELEMENT_CATEGORY_TEXT=$(cat "$TMP"/INLINE.log "$TMP"/STRUCTURE_GEOM.log \
  "$TMP"/TRANSPORT_GEOM.log "$TMP"/SCATRA_GEOM.log | grep -ci 'expected element category')"
exit 0
