#!/bin/bash

# Tier-2 for fourc::thermo_transient_mms#6 — before grading a dt series,
# check where 4C actually stopped.  The time loop ends at the last step that
# does not pass MAXTIME, so a dt that does not divide MAXTIME UNDERSHOOTS it,
# silently, and the levels of the series then hold solutions at DIFFERENT
# final times.  Errors measured against the exact solution at the requested
# time are then not a convergence study at all.
#
#   ALIGNED  dt = 0.1 / 0.05 / 0.025 , MAXTIME 0.4 -> all stop at 0.4
#   RAGGED   dt = 0.1 / 0.06 / 0.036 , MAXTIME 0.4 -> stop at 0.4 / 0.36 / 0.396
#
# NUMSTEP is 1000 in every run, so MAXTIME alone decides the stop.  4C prints
# the final time on its own 'Finalised:' line, which is the cheapest possible
# check and the one the entry tells you to make.
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

CENTRE=545                      # node at (0.5, 0.5) of the 32x32 mesh
# A dt series that all ends at the requested MAXTIME 0.4, and one that does
# not.  NUMSTEP is set far above what MAXTIME allows in every case, so it is
# MAXTIME that decides where the run stops.
for dt in 0.1 0.05 0.025; do
  mms "{\"n\":32,\"dt\":$dt,\"numstep\":1000,\"maxtime\":0.4,\"results\":[[$CENTRE,0.0,1e-30]]}" \
      "$TMP/aligned_$dt.yaml"
done
for dt in 0.1 0.06 0.036; do
  mms "{\"n\":32,\"dt\":$dt,\"numstep\":1000,\"maxtime\":0.4,\"results\":[[$CENTRE,0.0,1e-30]]}" \
      "$TMP/ragged_$dt.yaml"
done

final_time() {  # $1 = deck, $2 = tag -> echoes "TAG dt=... time=... value=..."
  run4c "$1" "$TMP/o_$2" > "$TMP/$2.log" 2>&1
  local ft; ft=$(grep '^Finalised:' "$TMP/$2.log" | tail -1 | grep -oP 'time \K[0-9.eE+-]+')
  local v;  v=$(grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/$2.log" | head -1)
  echo "$2 final_time=$ft value=$v"
}

for dt in 0.1 0.05 0.025; do final_time "$TMP/aligned_$dt.yaml" "ALIGNED_$dt"; done
for dt in 0.1 0.06 0.036; do final_time "$TMP/ragged_$dt.yaml"  "RAGGED_$dt";  done

# 4C echoes where it really stopped, and for the ragged series that is NOT
# the MAXTIME that was asked for.
grep -m1 -oP '^Finalised: step \d+ \| nstep 1000 \| time 0\.36 \| dt 0\.06.*' "$TMP/RAGGED_0.06.log"
grep -m1 -oP '^Finalised: step \d+ \| nstep 1000 \| time 0\.396 \| dt 0\.036.*' "$TMP/RAGGED_0.036.log"

python3 - "$TMP" <<'ANALYSIS'
import math, re, sys
from pathlib import Path
T = Path(sys.argv[1])
def read(tag):
    log = (T/f"{tag}.log").read_text()
    ft = float(re.findall(r"^Finalised:.*?time ([0-9.eE+-]+)", log, re.M)[-1])
    v = float(re.search(r"actresult=\s*([-0-9.e+]+)", log).group(1))
    return ft, v
def uex(t):   # centre node (0.5, 0.5)
    return 1 + (1.0 + 0.5*0.5)*math.cos(2*math.pi*t)

for label, dts in (("ALIGNED", ["0.1", "0.05", "0.025"]),
                   ("RAGGED",  ["0.1", "0.06", "0.036"])):
    times, errs = [], []
    for d in dts:
        ft, v = read(f"{label}_{d}")
        times.append(ft)
        errs.append(abs(v - uex(0.4)))     # graded against the REQUESTED time
    print(f"{label}_FINAL_TIMES = " + " ".join(f"{t:g}" for t in times))
    print(f"{label}_ERRORS      = " + " ".join(f"{e:.4e}" for e in errs))
    ratios = [errs[i]/errs[i+1] for i in range(len(errs)-1)]
    print(f"{label}_ERROR_RATIOS = " + " ".join(f"{r:.3f}" for r in ratios))
    same = max(times) - min(times) < 1e-12
    print(f"{label}_ALL_END_AT_THE_SAME_TIME=" + ("yes" if same else "no"))
    print(f"{label}_ERRORS_DECREASE_MONOTONICALLY=" +
          ("yes" if all(errs[i] > errs[i+1] for i in range(len(errs)-1)) else "no"))
ANALYSIS
exit 0
