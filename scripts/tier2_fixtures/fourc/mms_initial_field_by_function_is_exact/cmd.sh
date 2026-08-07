#!/bin/bash

# Tier-2 for fourc::thermo_transient_mms#2 — THERMAL DYNAMIC INITIALFIELD
# 'field_by_function' + INITFUNCNO evaluates the space-time function at t=0
# at the NODES, exactly.
#
# The probe reads the initial field directly instead of inferring it: with
# NUMSTEP 0 the time loop body never executes, so the RESULT DESCRIPTION
# tests the initial state and nothing else.  Two nodes are checked against
# the analytic u*(x,0) to 1e-14 —
#
#   node 41 at (0.5 , 0.5 ) : u* = 2.25
#   node 29 at (0.125, 0.375): u* = 1.4160533905932737
#
# and 'zero_field' is the control: the same two nodes come back exactly 0.
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

# The manufactured solution, its exact source, and the mesh, all emitted by
# one small generator so the arms differ ONLY in the knob under test:
#     u*(x,y,t) = 1 + (sin(pi x) sin(pi y) + 0.5 x) cos(2 pi t)
#     q         = rho c du*/dt - kappa lap(u*)     (FUNCT2)
# kx, ky and omega are baked in as numeric literals.
mms() {  # $1 = JSON knobs, $2 = output file
python3 - "$1" > "$2" <<'PY'
import sys, json, math
k = json.loads(sys.argv[1])
n        = k.get("n", 16)
dt       = k.get("dt", 0.05)
numstep  = k.get("numstep", 8)
maxtime  = k.get("maxtime", numstep * dt)
theta    = k.get("theta", 0.5)
neumann  = k.get("neumann", "DESIGN SURF NEUMANN CONDITIONS")
twice    = k.get("neumann_twice", False)
dfunct   = k.get("dirich_funct", 1)
initial  = k.get("initialfield", "field_by_function")
u_over   = k.get("u_override")
results  = k.get("results", [])          # list of [node, value, tol]
record   = k.get("record_all_nodes", False)
kappa = rho = c = 1.0
offset, amp, grad = 1.0, 1.0, 0.5
omega = 2.0 * math.pi
kx = ky = math.pi
spatial = f"({amp:.16g}*sin({kx:.16g}*x)*sin({ky:.16g}*y) + {grad:.16g}*x)"
f_t  = f"cos({omega:.16g}*t)"
fp_t = f"(-{omega:.16g}*sin({omega:.16g}*t))"
u = u_over if u_over else f"{offset:.16g} + {spatial}*{f_t}"
q = (f"{rho*c:.16g}*{spatial}*{fp_t}"
     f" + {kappa*amp*(kx*kx+ky*ky):.16g}*sin({kx:.16g}*x)*sin({ky:.16g}*y)*{f_t}")
ids, coords, nid = {}, [], 0
for j in range(n + 1):
    for i in range(n + 1):
        nid += 1
        ids[(i, j)] = nid
        coords.append(f"NODE {nid} COORD {i/n:.16g} {j/n:.16g} 0.0")
els = []
for j in range(n):
    for i in range(n):
        els.append(f"{len(els)+1} THERMO QUAD4 {ids[(i,j)]} {ids[(i+1,j)]} "
                   f"{ids[(i+1,j+1)]} {ids[(i,j+1)]} MAT 1")
boundary = sorted({ids[(i, j)] for j in range(n+1) for i in range(n+1)
                   if i in (0, n) or j in (0, n)})
if record:
    results = [[i, 0.0, 1e-30] for i in range(1, nid + 1)]
y = f'''PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: OneStepTheta
  INITIALFIELD: "{initial}"
  INITFUNCNO: 1
  TIMESTEP: {dt:.16g}
  NUMSTEP: {numstep}
  MAXTIME: {maxtime:.16g}
  RESULTSEVERY: {max(numstep, 1)}
  RESTARTEVERY: 0
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/ONESTEPTHETA:
  THETA: {theta:.16g}
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "T"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: {rho*c:.16g}
      CONDUCT:
        constant: [{kappa:.16g}]
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{u}"
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{q}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [{dfunct}]
'''
if neumann:
    y += f'''{neumann}:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
'''
    if twice:
        y += '''  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
'''
if results:
    y += "RESULT DESCRIPTION:\n"
    for node, val, tol in results:
        y += (f'  - THERMAL:\n      DIS: "thermo"\n      NODE: {node}\n'
              f'      QUANTITY: "temp"\n      VALUE: {val:.17g}\n'
              f'      TOLERANCE: {tol:.17g}\n')
y += "DLINE-NODE TOPOLOGY:\n"
y += "".join(f'  - "NODE {i} DLINE 1"\n' for i in boundary)
y += "DSURF-NODE TOPOLOGY:\n"
y += "".join(f'  - "NODE {i} DSURFACE 1"\n' for i in range(1, nid + 1))
if twice:
    y += "".join(f'  - "NODE {i} DSURFACE 2"\n' for i in range(1, nid + 1))
y += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in coords)
y += "THERMO ELEMENTS:\n" + "".join(f'  - "{s}"\n' for s in els)
sys.stdout.write(y)
PY
}

mms "{\"n\":8,\"numstep\":0,\"maxtime\":0,\"results\":[[41,2.25,1e-14],[29,1.4160533905932737,1e-14]]}" \
    "$TMP/by_function.yaml"
mms "{\"n\":8,\"numstep\":0,\"maxtime\":0,\"initialfield\":\"zero_field\",\"results\":[[41,2.25,1e-14],[29,1.4160533905932737,1e-14]]}" \
    "$TMP/zero_field.yaml"
# ...and a recording run so the actual nodal numbers appear in the evidence.
mms "{\"n\":8,\"numstep\":0,\"maxtime\":0,\"results\":[[41,0.0,1e-30],[29,0.0,1e-30]]}" \
    "$TMP/record.yaml"

probe BY_FUNCTION "$TMP/by_function.yaml"
probe ZERO_FIELD  "$TMP/zero_field.yaml"
probe RECORD      "$TMP/record.yaml"

echo "BY_FUNCTION_CORRECT=$(grep -c 'is CORRECT' "$TMP/BY_FUNCTION.log")"
grep -m1 -F "processor 0 finished normally" "$TMP/BY_FUNCTION.log"
# The recorded values are the function evaluated at t=0, to the last digit.
grep -m2 -F "is WRONG --> actresult=" "$TMP/RECORD.log"
# zero_field leaves both nodes at exactly zero.
grep -m1 -F "is WRONG --> actresult= 0.00000000000000000e+00" "$TMP/ZERO_FIELD.log"
echo "ZERO_FIELD_CORRECT=$(grep -c 'is CORRECT' "$TMP/ZERO_FIELD.log")"
exit 0
