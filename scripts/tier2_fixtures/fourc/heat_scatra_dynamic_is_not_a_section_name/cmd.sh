#!/bin/bash

# Tier-2 for fourc::heat#0 — the dynamics section is 'SCALAR TRANSPORT
# DYNAMIC' and nothing shorter, and the wrong name is a hard abort at parse
# rather than a silent fall-back to defaults.
#
# Probed on a 3-D HEX8 heat-conduction deck (phi = 1 on the x=0 face, 0 on
# the x=1 face, MAT_scatra DIFFUSIVITY 1, ten backward-Euler steps), which is
# the shape a beginner writes first:
#
#   BASE    'SCALAR TRANSPORT DYNAMIC' -> exit 0, result test CORRECT
#   ABBREV  'SCATRA DYNAMIC'           -> exit 1 before the time loop
#   TRUNC   'SCALAR TRANSPORT'         -> exit 1
#   TRANSPD 'TRANSPORT DYNAMIC'        -> exit 1
#
# All three rejections come from section-name validation in
# core/io/src/4C_io_input_file.cpp and name the offending section verbatim.
# The 'unknown section: SCATRA DYNAMIC' banner an earlier version of this
# entry quoted appears in none of the logs.
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
# ONLY in the section name under test.
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

NODE=22          # interior node at (1/3, 1/3, 1/3)
TRUE=0.595821728320196309

heat "{\"results\":[[$NODE,$TRUE,1e-12]]}"                  "$TMP/base.yaml"
heat "{\"dyn_section\":\"SCATRA DYNAMIC\"}"                  "$TMP/abbrev.yaml"
heat "{\"dyn_section\":\"SCALAR TRANSPORT\"}"                "$TMP/trunc.yaml"
heat "{\"dyn_section\":\"TRANSPORT DYNAMIC\"}"               "$TMP/transpd.yaml"

probe BASE    "$TMP/base.yaml"
probe ABBREV  "$TMP/abbrev.yaml"
probe TRUNC   "$TMP/trunc.yaml"
probe TRANSPD "$TMP/transpd.yaml"

# The full spelling is the one that works, all the way to a passing result test.
grep -m1 -F "is CORRECT" "$TMP/BASE.log"
grep -m1 -F "processor 0 finished normally" "$TMP/BASE.log"

# Every shortening is refused by name, before anything runs.
grep -m1 -F "Section 'SCATRA DYNAMIC' is not a valid section name." "$TMP/ABBREV.log"
grep -m1 -F "Section 'SCALAR TRANSPORT' is not a valid section name." "$TMP/TRUNC.log"
grep -m1 -F "Section 'TRANSPORT DYNAMIC' is not a valid section name." "$TMP/TRANSPD.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/ABBREV.log"

# "Hard abort BEFORE anything runs" is the half that distinguishes this from a
# silent fall-back: the abbreviated deck never prints a single time step.
echo "BASE_TIME_STEPS=$(grep -c 'STEP = ' "$TMP/BASE.log")"
echo "ABBREV_TIME_STEPS=$(grep -c 'STEP = ' "$TMP/ABBREV.log")"

# The parser banner an earlier version of this entry quoted is nowhere.
echo "CLAIMED_UNKNOWN_SECTION_TEXT=$(grep -ci 'unknown section' "$TMP/ABBREV.log")"
exit 0
