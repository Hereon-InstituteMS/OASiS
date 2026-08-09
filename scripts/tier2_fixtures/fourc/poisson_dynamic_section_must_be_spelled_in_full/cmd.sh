#!/bin/bash

# Tier-2 for fourc::poisson#0 — the dynamics section is 'SCALAR TRANSPORT
# DYNAMIC' spelled out, and no near miss falls back to defaults.
#
# Probed on the smallest thing that deserves the name Poisson: a unit square
# of TRANSP TRI3 with TIMEINTEGR "Stationary", phi = 1 on the left edge and 0
# on the right, whose exact answer at the centre node is 1/2.
#
#   BASE     'SCALAR TRANSPORT DYNAMIC'  -> exit 0, result test CORRECT
#   ABBREV   'SCATRA DYNAMIC'            -> exit 1 before the solve
#   PLURAL   'SCALAR TRANSPORT DYNAMICS' -> exit 1
#   POISSONS 'POISSON DYNAMIC'           -> exit 1
#
# All three rejections come from section-name validation in
# core/io/src/4C_io_input_file.cpp and quote the offending name.
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

NODE=13          # centre of the 4x4 square, at (0.5, 0.5)
TRUE=0.500000000000000111

pois "{\"results\":[[$NODE,$TRUE,1e-12]]}"                    "$TMP/base.yaml"
pois "{\"dyn_section\":\"SCATRA DYNAMIC\"}"                    "$TMP/abbrev.yaml"
pois "{\"dyn_section\":\"SCALAR TRANSPORT DYNAMICS\"}"         "$TMP/plural.yaml"
pois "{\"dyn_section\":\"POISSON DYNAMIC\"}"                   "$TMP/poissons.yaml"

probe BASE     "$TMP/base.yaml"
probe ABBREV   "$TMP/abbrev.yaml"
probe PLURAL   "$TMP/plural.yaml"
probe POISSONS "$TMP/poissons.yaml"

# The full spelling solves the problem and the answer is the exact one.
grep -m1 -F "is CORRECT" "$TMP/BASE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"
grep -m1 -F "Stationary    STEP = 1/1" "$TMP/BASE.log"

# Every near miss is refused by name.
grep -m1 -F "Section 'SCATRA DYNAMIC' is not a valid section name." "$TMP/ABBREV.log"
grep -m1 -F "Section 'SCALAR TRANSPORT DYNAMICS' is not a valid section name." "$TMP/PLURAL.log"
grep -m1 -F "Section 'POISSON DYNAMIC' is not a valid section name." "$TMP/POISSONS.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/POISSONS.log"

# Nothing runs: the abbreviated deck never reaches the one stationary step.
echo "BASE_SOLVE_STEPS=$(grep -c 'STEP = ' "$TMP/BASE.log")"
echo "ABBREV_SOLVE_STEPS=$(grep -c 'STEP = ' "$TMP/ABBREV.log")"
echo "CLAIMED_UNKNOWN_SECTION_TEXT=$(grep -ci 'unknown section' "$TMP/ABBREV.log")"
exit 0
