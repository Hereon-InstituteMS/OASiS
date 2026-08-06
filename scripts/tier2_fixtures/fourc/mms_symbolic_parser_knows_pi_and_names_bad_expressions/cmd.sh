#!/bin/bash

# Tier-2 for fourc::thermo_transient_mms#5 — what the 4C symbolic-expression
# parser knows, and what it says when an expression is wrong.
#
#   pi          IS a symbol.  The manufactured solution written with 'pi' and
#               '2*pi' reproduces the version with the constants baked in as
#               numeric literals BIT FOR BIT, so writing 'pi' costs nothing.
#   bad syntax  fails at input read, echoing the expression with a caret under
#               the offending character and naming what it wanted, from
#               core/utils/src/functions/4C_utils_symbolic_expression.cpp.
#   bad symbol  is a DIFFERENT and later failure: the expression parses, the
#               discretisation is built, and only the first evaluation reports
#               "Missing variables foo to evaluate expression '...'".
#
# The initial field is read with NUMSTEP 0 so the RESULT DESCRIPTION reports
# the function evaluation itself rather than a time-stepped consequence.
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

# 1. 'pi' is a real symbol: the SAME manufactured solution written with 'pi'
#    instead of the baked-in literal must reproduce the initial field to the
#    last digit.  NUMSTEP 0 so the test reads the initial field itself.
LITERAL='1 + (1*sin(3.141592653589793*x)*sin(3.141592653589793*y) + 0.5*x)*cos(6.283185307179586*t)'
WITH_PI='1 + (1*sin(pi*x)*sin(pi*y) + 0.5*x)*cos(2*pi*t)'

mms "{\"n\":8,\"numstep\":0,\"maxtime\":0,\"u_override\":\"$LITERAL\",\"results\":[[41,0.0,1e-30],[29,0.0,1e-30]]}" \
    "$TMP/literal.yaml"
mms "{\"n\":8,\"numstep\":0,\"maxtime\":0,\"u_override\":\"$WITH_PI\",\"results\":[[41,0.0,1e-30],[29,0.0,1e-30]]}" \
    "$TMP/with_pi.yaml"
# 2. A syntactically broken expression.
mms "{\"n\":8,\"numstep\":0,\"maxtime\":0,\"u_override\":\"1.0 + sin(pi*x\"}" \
    "$TMP/malformed.yaml"
# 3. A syntactically fine expression naming a symbol that does not exist.
mms "{\"n\":8,\"numstep\":0,\"maxtime\":0,\"u_override\":\"1.0 + foo*x\"}" \
    "$TMP/unknown_symbol.yaml"

probe LITERAL        "$TMP/literal.yaml"
probe WITH_PI        "$TMP/with_pi.yaml"
probe MALFORMED      "$TMP/malformed.yaml"
probe UNKNOWN_SYMBOL "$TMP/unknown_symbol.yaml"

grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/LITERAL.log" | sort > "$TMP/a.txt"
grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/WITH_PI.log" | sort > "$TMP/b.txt"
echo "PI_VALUES_READ=$(wc -l < "$TMP/b.txt")"
if [ -s "$TMP/a.txt" ] && cmp -s "$TMP/a.txt" "$TMP/b.txt"; then
  echo "PI_IS_BIT_IDENTICAL_TO_LITERAL=yes"
else
  echo "PI_IS_BIT_IDENTICAL_TO_LITERAL=no"
fi
grep -m1 -F "is WRONG --> actresult= 2.25000000000000000e+00" "$TMP/WITH_PI.log"

# The syntax error names the expression, points at the offending character
# and says what it wanted — at input read, before any discretisation exists.
grep -m1 -F "Error while parsing:" "$TMP/MALFORMED.log"
grep -m1 -F "1.0 + sin(pi*x" "$TMP/MALFORMED.log"
grep -m1 -F "Expected closing parenthesis." "$TMP/MALFORMED.log"
grep -m1 -F "4C_utils_symbolic_expression.cpp" "$TMP/MALFORMED.log"
echo "MALFORMED_REACHED_DISCRETISATION=$(grep -c 'fill_complete() on discretization thermo' "$TMP/MALFORMED.log")"

# An unknown SYMBOL is a different failure and arrives later: the expression
# parses, and only the first evaluation notices the variable is missing.
grep -m1 -F "Missing variables foo to evaluate expression '1.0 + foo*x'" "$TMP/UNKNOWN_SYMBOL.log"
grep -m1 -F "4C_utils_symbolic_expression_details.hpp" "$TMP/UNKNOWN_SYMBOL.log"
if [ "$(grep -c 'fill_complete() on discretization thermo' "$TMP/UNKNOWN_SYMBOL.log")" -gt 0 ]; then
  echo "UNKNOWN_SYMBOL_REACHED_DISCRETISATION=yes"
else
  echo "UNKNOWN_SYMBOL_REACHED_DISCRETISATION=no"
fi
exit 0
