#!/bin/bash

# Tier-2 for fourc::heat#5 — BDF2 needs no hand-written start-up, and 4C
# gives you no way to see the one it does for you.
#
# The same 3-D heat deck under TIMEINTEGR "BDF2" and under "One_Step_Theta"
# with THETA 1.0 (which IS backward Euler), for 1, 2 and 3 steps:
#
#   1 step  -> the two agree to the last printed digit
#   2 steps -> they differ
#   3 steps -> they differ
#
# The later steps are what stop the first observation being vacuous: if BDF2
# never engaged, every step would agree.  And the header line 4C prints for
# step 1 of the BDF2 run says BDF2, exactly like every other step — the
# lower-order start-up is announced nowhere.
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
value() {  # $1 = label -> the recorded nodal value
  grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/$1.log" | head -1
}
for ns in 1 2 3; do
  heat "{\"timeintegr\":\"BDF2\",\"numstep\":$ns,\"dt\":0.02,\"results\":[[$NODE,0.0,1e-30]]}" \
       "$TMP/bdf2_$ns.yaml"
  heat "{\"timeintegr\":\"One_Step_Theta\",\"numstep\":$ns,\"dt\":0.02,\"results\":[[$NODE,0.0,1e-30]]}" \
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
# From step 2 the schemes part company, or the first result would be vacuous.
if [ "$(value BDF2_2)" != "$(value BE_2)" ] && [ "$(value BDF2_3)" != "$(value BE_3)" ]; then
  echo "LATER_STEPS_DIFFER=yes"
else
  echo "LATER_STEPS_DIFFER=no"
fi

# The step header calls it BDF2 on the very step that is not BDF2.
grep -m1 -F "BDF2       STEP = 1/3" "$TMP/BDF2_3.log"
echo "BDF2_BANNER_STEPS=$(grep -c 'BDF2       STEP' "$TMP/BDF2_3.log")"
echo "STARTUP_ANNOUNCED=$(grep -ciE 'start-?up|backward euler' "$TMP/BDF2_3.log")"
exit 0
