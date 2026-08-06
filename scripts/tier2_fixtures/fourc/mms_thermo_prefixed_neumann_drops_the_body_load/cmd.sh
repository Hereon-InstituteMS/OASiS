#!/bin/bash
# Tier-2 for fourc::thermo_transient_mms#0 — in a standalone Thermo run the
# volumetric heat source must sit in the PLAIN 'DESIGN SURF NEUMANN
# CONDITIONS'.  The THERMO-prefixed section parses, is accepted, and is then
# never looked at: the source is dropped, the run still exits 0 from the
# solver's point of view, and nothing in the log says so.
#
# The probe is a manufactured solution
#     u*(x,y,t) = 1 + (sin(pi x) sin(pi y) + 0.5 x) cos(2 pi t)
# with the matching source q = rho c du*/dt - kappa lap(u*) as FUNCT2, u* as
# time-dependent Dirichlet on the whole boundary and as the initial field.
# The centre node is result-tested against the analytic value with a
# tolerance far looser than the mesh's own discretisation error, so only a
# structurally wrong right-hand side can fail it.
#
# plain section  -> centre node lands within tolerance, 'is CORRECT', exit 0
# THERMO-prefixed-> centre node is off by ~0.9, 'is WRONG', exit 1
#
# and crucially the prefixed arm produces NO 'not a valid section name' and
# no warning of any kind.  This extends the recorded THERMO-prefixed DIRICH
# finding (fourc::thermal#3) to the Neumann/body-load path.
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

mms() {  # $1 = Neumann section name, $2 = output file
python3 - "$1" > "$2" <<'PY'
import sys, math
neumann = sys.argv[1]
n, dt, numstep, theta = 16, 0.05, 8, 0.5
kappa = rho = c = 1.0
temp_offset, amp, grad = 1.0, 1.0, 0.5
omega = 2.0 * math.pi
kx = ky = math.pi
spatial = f"({amp:.16g}*sin({kx:.16g}*x)*sin({ky:.16g}*y) + {grad:.16g}*x)"
f_t = f"cos({omega:.16g}*t)"
fp_t = f"(-{omega:.16g}*sin({omega:.16g}*t))"
u = f"{temp_offset:.16g} + {spatial}*{f_t}"
q = (f"{rho*c:.16g}*{spatial}*{fp_t}"
     f" + {kappa*amp*(kx*kx+ky*ky):.16g}*sin({kx:.16g}*x)*sin({ky:.16g}*y)*{f_t}")
ids = {}
coords = []
nid = 0
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
centre = ids[(n//2, n//2)]
t_end = numstep * dt
exact = temp_offset + (amp*math.sin(kx*0.5)*math.sin(ky*0.5)
                       + grad*0.5)*math.cos(omega*t_end)
out = [f'''PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: OneStepTheta
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: {dt:.16g}
  NUMSTEP: {numstep}
  MAXTIME: {t_end:.16g}
  RESULTSEVERY: {numstep}
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
    FUNCT: [1]
{neumann}:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
RESULT DESCRIPTION:
  - THERMAL:
      DIS: "thermo"
      NODE: {centre}
      QUANTITY: "temp"
      VALUE: {exact:.17g}
      TOLERANCE: 5e-3
DLINE-NODE TOPOLOGY:''']
out += [f'  - "NODE {i} DLINE 1"' for i in boundary]
out += ["DSURF-NODE TOPOLOGY:"]
out += [f'  - "NODE {i} DSURFACE 1"' for i in range(1, nid + 1)]
out += ["NODE COORDS:"] + [f'  - "{s}"' for s in coords]
out += ["THERMO ELEMENTS:"] + [f'  - "{s}"' for s in els]
print("\n".join(out))
PY
}

mms "DESIGN SURF NEUMANN CONDITIONS"        "$TMP/plain.yaml"
mms "DESIGN SURF THERMO NEUMANN CONDITIONS" "$TMP/prefixed.yaml"

probe PLAIN    "$TMP/plain.yaml"
probe PREFIXED "$TMP/prefixed.yaml"

grep -m1 -F "is CORRECT" "$TMP/PLAIN.log"
grep -m1 -F "processor 0 finished normally" "$TMP/PLAIN.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/PREFIXED.log"

# The prefixed section is a VALID section name — that is what makes it a trap.
echo "PREFIXED_SECTION_NAME_REJECTED=$(grep -c 'not a valid section name' "$TMP/PREFIXED.log")"
# ...and 4C never mentions that the condition was not applied.
echo "PREFIXED_NEUMANN_WARNINGS=$(grep -ciE 'neumann.*(ignor|unus|not applied|no effect|dropped)' "$TMP/PREFIXED.log")"
# The prefixed run reaches the very end of the time loop; only the result
# test catches it.
echo "PREFIXED_COMPLETED_TIME_LOOP=$(grep -c 'Finalised: step 8' "$TMP/PREFIXED.log")"
exit 0
