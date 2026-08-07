#!/bin/bash

# Tier-2 for fourc::poisson#5 — the automatic BDF2 start-up, pinned by an
# exact equality and TWO controls rather than an error table.
#
# The same TRI3 deck, run transiently for 1, 2 and 4 steps under three
# settings: TIMEINTEGR "BDF2", "One_Step_Theta" with THETA 1.0 (backward
# Euler) and "One_Step_Theta" with THETA 0.5 (Crank-Nicolson).
#
#   1 step  -> BDF2 equals backward Euler to the last printed digit, and does
#              NOT equal Crank-Nicolson.  The second half matters: without it
#              the equality could be an artefact of a start-up that agrees
#              with everything.
#   2 and 4 -> BDF2 parts company with backward Euler, so BDF2 really engages
#              after the first step.
#
# No hand-rolled start-up is needed, and 4C announces none of this.
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
value() {  # $1 = label -> the recorded nodal value
  grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/$1.log" | head -1
}
for ns in 1 2 4; do
  pois "{\"timeintegr\":\"BDF2\",\"numstep\":$ns,\"dt\":0.005,\"results\":[[$NODE,0.0,1e-30]]}" \
       "$TMP/bdf2_$ns.yaml"
  pois "{\"timeintegr\":\"One_Step_Theta\",\"theta\":1.0,\"numstep\":$ns,\"dt\":0.005,\"results\":[[$NODE,0.0,1e-30]]}" \
       "$TMP/be_$ns.yaml"
  pois "{\"timeintegr\":\"One_Step_Theta\",\"theta\":0.5,\"numstep\":$ns,\"dt\":0.005,\"results\":[[$NODE,0.0,1e-30]]}" \
       "$TMP/cn_$ns.yaml"
  probe "BDF2_$ns" "$TMP/bdf2_$ns.yaml"
  probe "BE_$ns"   "$TMP/be_$ns.yaml"
  probe "CN_$ns"   "$TMP/cn_$ns.yaml"
  echo "STEPS_$ns BDF2=$(value "BDF2_$ns") THETA1=$(value "BE_$ns") THETA05=$(value "CN_$ns")"
done

grep -m1 -F "is WRONG --> actresult=" "$TMP/BDF2_1.log"

if [ "$(value BDF2_1)" = "$(value BE_1)" ] && [ -n "$(value BDF2_1)" ]; then
  echo "FIRST_STEP_IS_BACKWARD_EULER=yes"
else
  echo "FIRST_STEP_IS_BACKWARD_EULER=no"
fi
# The first step is backward Euler SPECIFICALLY, not merely 'some one-step
# scheme': it does not match theta = 1/2.
if [ "$(value BDF2_1)" != "$(value CN_1)" ] && [ -n "$(value CN_1)" ]; then
  echo "FIRST_STEP_IS_NOT_CRANK_NICOLSON=yes"
else
  echo "FIRST_STEP_IS_NOT_CRANK_NICOLSON=no"
fi
# And BDF2 does engage afterwards, or the first equality would be vacuous.
if [ "$(value BDF2_2)" != "$(value BE_2)" ] && [ "$(value BDF2_4)" != "$(value BE_4)" ]; then
  echo "LATER_STEPS_DIFFER=yes"
else
  echo "LATER_STEPS_DIFFER=no"
fi

grep -m1 -F "BDF2       STEP = 1/4" "$TMP/BDF2_4.log"
echo "BDF2_BANNER_STEPS=$(grep -c 'BDF2       STEP' "$TMP/BDF2_4.log")"
echo "STARTUP_ANNOUNCED=$(grep -ciE 'start-?up|backward euler' "$TMP/BDF2_4.log")"
exit 0
