#!/bin/bash

# Tier-2 for fourc::scalar_transport#5 — BDF2 needs no hand-written start-up
# because 4C already does one backward-Euler step, and the cleanest possible
# evidence is an exact equality rather than an error table.
#
# Run the same deck for 1, 2 and 4 steps under TIMEINTEGR "BDF2" and under
# "One_Step_Theta" with THETA 1.0 (which IS backward Euler):
#
#   1 step  -> the two agree to the last printed digit
#   2 steps -> they differ
#   4 steps -> they differ
#
# The first line is the claim.  The other two are what stop it being vacuous:
# if BDF2 simply never engaged, every step would agree.  4C announces none of
# this (STARTUP_ANNOUNCED=0); the switch is visible only in the numbers.
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

NODE=8
value() {  # $1 = label -> the recorded nodal value
  grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/$1.log" | head -1
}
for ns in 1 2 4; do
  sca "{\"timeintegr\":\"BDF2\",\"numstep\":$ns,\"dt\":0.01,\"results\":[[$NODE,0.0,1e-30]]}" \
      "$TMP/bdf2_$ns.yaml"
  sca "{\"timeintegr\":\"One_Step_Theta\",\"numstep\":$ns,\"dt\":0.01,\"results\":[[$NODE,0.0,1e-30]]}" \
      "$TMP/be_$ns.yaml"
  probe "BDF2_$ns" "$TMP/bdf2_$ns.yaml"
  probe "BE_$ns"   "$TMP/be_$ns.yaml"
  echo "STEPS_$ns BDF2=$(value "BDF2_$ns") BACKWARD_EULER=$(value "BE_$ns")"
done

grep -m1 -F "is WRONG --> actresult=" "$TMP/BDF2_1.log"

# One step of BDF2 must BE one step of backward Euler, bit for bit.
if [ "$(value BDF2_1)" = "$(value BE_1)" ] && [ -n "$(value BDF2_1)" ]; then
  echo "FIRST_STEP_IS_BACKWARD_EULER=yes"
else
  echo "FIRST_STEP_IS_BACKWARD_EULER=no"
fi
# From step 2 the schemes must part company, or the first result would be
# vacuous (BDF2 silently never engaging).
if [ "$(value BDF2_2)" != "$(value BE_2)" ] && [ "$(value BDF2_4)" != "$(value BE_4)" ]; then
  echo "LATER_STEPS_DIFFER=yes"
else
  echo "LATER_STEPS_DIFFER=no"
fi
# 4C never announces the start-up switch.
echo "STARTUP_ANNOUNCED=$(grep -ciE 'start-?up|backward euler' "$TMP/BDF2_4.log")"
exit 0
