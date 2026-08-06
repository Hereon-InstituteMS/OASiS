#!/bin/bash

# Tier-2 for fourc::thermo_transient_mms#4 — on a FIXED mesh the error
# against the exact MMS solution stops measuring the time integrator once it
# reaches the spatial Q1 floor, and a dt-halving table built from it reports
# nonsense.  Richardson differences ||u_dt - u_dt/2|| taken on the SAME mesh
# cancel that floor exactly and recover the temporal order.
#
# Every nodal temperature is read straight out of 4C: the deck carries a
# RESULT DESCRIPTION entry on every node with an impossible target, so 4C
# prints "actresult=" for each one and the script below reads the field off
# its own log.  No post-processing library, no VTK parsing.
#
# Six dt levels on one 32x32 mesh at theta = 0.5, then a short three-level
# series at theta = 1.0 as the control that Richardson is measuring the
# SCHEME rather than returning 2 by construction.
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

NODES=1089            # (32+1)^2
run_series() {  # $1 = theta, $2 = tag, rest = dt list
  local theta="$1" tag="$2"; shift 2
  for dt in "$@"; do
    local ns; ns=$(python3 -c "print(round(0.4/$dt))")
    mms "{\"n\":32,\"dt\":$dt,\"numstep\":$ns,\"theta\":$theta,\"record_all_nodes\":true}" \
        "$TMP/s.yaml"
    run4c "$TMP/s.yaml" "$TMP/o_s" > "$TMP/${tag}_$dt.log" 2>&1
    grep -oP 'at node\s+\K[0-9]+(?=.*actresult=)|actresult=\s*\K[-0-9.e+]+' \
         "$TMP/${tag}_$dt.log" > "$TMP/${tag}_$dt.vals"
    echo "SERIES $tag dt=$dt nodes_read=$(($(wc -l < "$TMP/${tag}_$dt.vals") / 2))"
  done
}

run_series 0.5 th05 0.1 0.05 0.025 0.0125 0.00625 0.003125
run_series 1.0 th10 0.025 0.0125 0.00625 0.003125

# 4C really did evaluate and report every node.
grep -m1 -F "Checking results of $NODES tests:" "$TMP/th05_0.1.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/th05_0.1.log"

python3 - "$TMP" <<'ANALYSIS'
import math, sys
from pathlib import Path
T = Path(sys.argv[1]); n = 32; t_end = 0.4
def uex(x, y, t):
    return 1 + (math.sin(math.pi*x)*math.sin(math.pi*y) + 0.5*x)*math.cos(2*math.pi*t)
coords, nid = {}, 0
for j in range(n+1):
    for i in range(n+1):
        nid += 1
        coords[nid] = (i/n, j/n)
def load(tag, dt):
    w = (T/f"{tag}_{dt}.vals").read_text().split()
    return {int(w[k]): float(w[k+1]) for k in range(0, len(w), 2)}
def rms_exact(d):
    return math.sqrt(sum((v - uex(*coords[k], t_end))**2 for k, v in d.items())/len(d))
def rms_diff(a, b):
    return math.sqrt(sum((a[k]-b[k])**2 for k in a)/len(a))
def orders(seq):
    return [math.log(seq[i]/seq[i+1], 2) for i in range(len(seq)-1)]

dts05 = ["0.1", "0.05", "0.025", "0.0125", "0.00625", "0.003125"]
s05 = [load("th05", d) for d in dts05]
e = [rms_exact(x) for x in s05]
r = [rms_diff(s05[i], s05[i+1]) for i in range(len(s05)-1)]
oe, orr = orders(e), orders(r)
print("RMS_VS_EXACT    = " + " ".join(f"{x:.4e}" for x in e))
print("ORDERS_VS_EXACT = " + " ".join(f"{x:+.3f}" for x in oe))
print("RICHARDSON_DIFF = " + " ".join(f"{x:.4e}" for x in r))
print("RICHARDSON_ORDERS_THETA_HALF = " + " ".join(f"{x:.3f}" for x in orr))

dts10 = ["0.025", "0.0125", "0.00625", "0.003125"]
s10 = [load("th10", d) for d in dts10]
o10 = orders([rms_diff(s10[i], s10[i+1]) for i in range(len(s10)-1)])
print("RICHARDSON_ORDERS_THETA_ONE  = " + " ".join(f"{x:.3f}" for x in o10))

print("RICHARDSON_RECOVERS_ORDER_2=" +
      ("yes" if all(1.9 <= x <= 2.1 for x in orr) else "no"))
print("RICHARDSON_SEPARATES_THE_SCHEMES=" +
      ("yes" if all(0.85 <= x <= 1.15 for x in o10) else "no"))
print("ERROR_VS_EXACT_SATURATES=" +
      ("yes" if abs(e[-1]-e[-2])/e[-1] < 0.10 else "no"))
print("ERROR_VS_EXACT_ORDER_MISLEADING=" +
      ("yes" if any(not (1.5 <= x <= 2.5) for x in oe) else "no"))
ANALYSIS
exit 0
