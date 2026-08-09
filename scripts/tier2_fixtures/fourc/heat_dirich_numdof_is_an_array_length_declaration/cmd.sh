#!/bin/bash

# Tier-2 for fourc::heat#4 — NUMDOF declares how long ONOFF/VAL/FUNCT are and
# is checked against THEM, not against the number of transported scalars.
#
#   NUMDOF1  NUMDOF 1, one-entry arrays                -> runs
#   NUMDOF2  NUMDOF 2, two-entry arrays, second VAL 7  -> ALSO runs, and gives
#            the IDENTICAL nodal value.  The surplus entry addresses a DOF the
#            scatra field does not have and is dropped without a word.
#   RAGGED   NUMDOF 2 with a one-entry VAL             -> refused, and 4C names
#            the array that is the wrong length
#
# The two phrasings an earlier version of this entry quoted, 'array size
# mismatch' and 'expected NUMDOF entries', are asserted absent.
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

NODE=22
heat "{\"results\":[[$NODE,0.0,1e-30]]}" "$TMP/numdof1.yaml"
heat "{\"numdof\":2,\"onoff\":\"[1, 1]\",\"val\":\"[1.0, 7.0]\",\"funct\":\"[1, 0]\",\"results\":[[$NODE,0.0,1e-30]]}" \
     "$TMP/numdof2.yaml"
heat "{\"numdof\":2,\"onoff\":\"[1, 1]\",\"val\":\"[1.0]\",\"funct\":\"[1, 0]\",\"results\":[[$NODE,0.0,1e-30]]}" \
     "$TMP/ragged.yaml"

probe NUMDOF1 "$TMP/numdof1.yaml"
probe NUMDOF2 "$TMP/numdof2.yaml"
probe RAGGED  "$TMP/ragged.yaml"

A=$(grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/NUMDOF1.log" | head -1)
B=$(grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/NUMDOF2.log" | head -1)
echo "NUMDOF1_VALUE=$A"
echo "NUMDOF2_VALUE=$B"
if [ -n "$A" ] && [ "$A" = "$B" ]; then
  echo "OVERDECLARED_NUMDOF_CHANGES_THE_ANSWER=no"
else
  echo "OVERDECLARED_NUMDOF_CHANGES_THE_ANSWER=yes"
fi
# The over-declared deck draws no complaint of its own: the single 'PROC 0
# ERROR in' line it prints is the deliberate recording result test.
echo "NUMDOF2_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/NUMDOF2.log")"

# What IS enforced is array length == NUMDOF, and 4C names the offender.
grep -m1 -F "Failed to match condition specification in section 'DESIGN SURF DIRICH CONDITIONS'." "$TMP/RAGGED.log"
grep -m1 -F "4C_fem_condition_definition.cpp" "$TMP/RAGGED.log"
grep -m1 -F "Candidate parameter 'VAL' has incorrect size" "$TMP/RAGGED.log"

# The two phrasings the entry used to quote appear nowhere.
echo "CLAIMED_ARRAY_SIZE_MISMATCH=$(grep -ci 'array size mismatch' "$TMP/RAGGED.log")"
echo "CLAIMED_EXPECTED_NUMDOF_ENTRIES=$(grep -ci 'expected NUMDOF entries' "$TMP/RAGGED.log")"
exit 0
