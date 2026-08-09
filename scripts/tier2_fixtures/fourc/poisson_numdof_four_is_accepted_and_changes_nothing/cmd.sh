#!/bin/bash

# Tier-2 for fourc::poisson#4 — NUMDOF is a declaration of how long
# ONOFF/VAL/FUNCT are, checked against THEM and not against the one scalar the
# Poisson problem transports.
#
#   NUMDOF1  NUMDOF 1, one-entry arrays                         -> runs
#   NUMDOF4  NUMDOF 4, four-entry arrays VAL [1, 2, 3, 4]       -> ALSO runs,
#            and returns the IDENTICAL centre value.  Three surplus entries
#            address DOFs the field does not have and vanish without comment.
#   RAGGED   NUMDOF 1 with a two-entry FUNCT                    -> refused,
#            and 4C names FUNCT as the array of the wrong size
#
# So over-declaring is not an error; disagreeing with your own declaration is.
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

NODE=13
pois "{\"results\":[[$NODE,0.0,1e-30]]}" "$TMP/numdof1.yaml"
pois "{\"numdof\":4,\"onoff\":\"[1, 1, 1, 1]\",\"val\":\"[1.0, 2.0, 3.0, 4.0]\",\"funct\":\"[1, 0, 0, 0]\",\"results\":[[$NODE,0.0,1e-30]]}" \
     "$TMP/numdof4.yaml"
pois "{\"numdof\":1,\"onoff\":\"[1]\",\"val\":\"[1.0]\",\"funct\":\"[1, 0]\",\"results\":[[$NODE,0.0,1e-30]]}" \
     "$TMP/ragged.yaml"

probe NUMDOF1 "$TMP/numdof1.yaml"
probe NUMDOF4 "$TMP/numdof4.yaml"
probe RAGGED  "$TMP/ragged.yaml"

A=$(grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/NUMDOF1.log" | head -1)
B=$(grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/NUMDOF4.log" | head -1)
echo "NUMDOF1_VALUE=$A"
echo "NUMDOF4_VALUE=$B"
if [ -n "$A" ] && [ "$A" = "$B" ]; then
  echo "OVERDECLARED_NUMDOF_CHANGES_THE_ANSWER=no"
else
  echo "OVERDECLARED_NUMDOF_CHANGES_THE_ANSWER=yes"
fi
# The only 'PROC 0 ERROR in' the over-declared deck prints is the deliberate
# recording result test.
echo "NUMDOF4_ERROR_LINES=$(grep -c 'PROC 0 ERROR in' "$TMP/NUMDOF4.log")"

# What IS enforced, and the array it names.
grep -m1 -F "Failed to match condition specification in section 'DESIGN LINE DIRICH CONDITIONS'." "$TMP/RAGGED.log"
grep -m1 -F "4C_fem_condition_definition.cpp" "$TMP/RAGGED.log"
grep -m1 -F "Candidate parameter 'FUNCT' has incorrect size" "$TMP/RAGGED.log"

echo "CLAIMED_ARRAY_SIZE_MISMATCH=$(grep -ci 'array size mismatch' "$TMP/RAGGED.log")"
echo "CLAIMED_EXPECTED_NUMDOF_ENTRIES=$(grep -ci 'expected NUMDOF entries' "$TMP/RAGGED.log")"
exit 0
