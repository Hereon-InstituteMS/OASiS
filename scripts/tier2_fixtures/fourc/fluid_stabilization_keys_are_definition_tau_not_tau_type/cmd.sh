#!/bin/bash

# Tier-2 for fourc::fluid#1, and a FALSIFICATION of every name in it.
#
# The entry says the stabilisation knobs live in 'FLUID DYNAMIC/STABILIZATION'
# and that the tau parameter is raised through 'TAU_TYPE / TAU_DEF'.  None of
# those three names exists, and the flag it calls 'GRAD-DIV' is spelled
# GRAD_DIV.  Measured on a self-contained 2-D deck at DYNVISCOSITY 1e-3:
#
#   DEFAULT    residual-based stabilisation as it comes -> runs
#   NOSTAB     STABTYPE "no_stabilization"              -> runs, DIFFERENT
#                                                          nodal velocity
#   TAUOK      DEFINITION_TAU "Taylor_Hughes_Zarins" in
#              'FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION'
#                                                       -> accepted, and a
#                                                          third answer again
#   STABSEC    'FLUID DYNAMIC/STABILIZATION'            -> not a section
#   TAUTYPE    TAU_TYPE                                 -> not a key
#   TAUDEF     TAU_DEF                                  -> not a key
#   GRADDIV    'GRAD-DIV' with a hyphen                 -> not a key
#
# So the rule survives — stabilisation choice really does move the answer, and
# the tau definition really is the knob — but every identifier the entry gives
# for it is wrong.  The Signal is wrong too: a 4C fluid run reports no
# integrated kinetic energy at all, so it cannot be watched growing.
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

# One generator emits the whole 2-D fluid deck, so the arms differ ONLY in the
# knob under test: a unit square of FLUID QUAD4 with MAT_fluid, u = 1 driven
# in on the left edge, no-slip top and bottom, pressure pinned at one corner.
fluid() {  # $1 = JSON knobs, $2 = output file
python3 - "$1" > "$2" <<'FLUIDPY'
import sys, json
k = json.loads(sys.argv[1])
n        = k.get("n", 6)
dynsect  = k.get("dyn_section", "FLUID DYNAMIC")
elesect  = k.get("element_section", "FLUID ELEMENTS")
visc     = k.get("visc", 0.01)
timeint  = k.get("timeintegr", "One_Step_Theta")
dt       = k.get("dt", 0.1)
numstep  = k.get("numstep", 5)
stab     = k.get("stab_section", "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION")
stabbody = k.get("stab_body", "")
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
        els.append(f"{len(els)+1} FLUID QUAD4 {ids[(i,j)]} {ids[(i+1,j)]} "
                   f"{ids[(i+1,j+1)]} {ids[(i,j+1)]} MAT 1 NA Euler")
left   = [ids[(0, j)] for j in range(1, n)]
top    = [ids[(i, n)] for i in range(n + 1)]
bottom = [ids[(i, 0)] for i in range(n + 1)]
corner = ids[(n, 0)]
stabblk = (stab + ":\n" + stabbody) if stabbody else ""
y = f'''PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Fluid"
{dynsect}:
  LINEAR_SOLVER: 1
  TIMEINTEGR: "{timeint}"
  THETA: 1.0
  TIMESTEP: {dt:.16g}
  NUMSTEP: {numstep}
  MAXTIME: {dt*numstep:.16g}
  ITEMAX: 20
  INITIALFIELD: "zero_field"
{stabblk}SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Fluid_Solver"
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: {visc:.16g}
      DENSITY: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "1.0"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1.0, 0.0, 0.0]
    FUNCT: [1, 0, 0]
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 3
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
{extra}'''
if results:
    y += "RESULT DESCRIPTION:\n"
    for node, q, v, tol in results:
        y += (f'  - FLUID:\n      DIS: "fluid"\n      NODE: {node}\n'
              f'      QUANTITY: "{q}"\n      VALUE: {v:.17g}\n'
              f'      TOLERANCE: {tol:.17g}\n')
y += "DNODE-NODE TOPOLOGY:\n" + f'  - "NODE {corner} DNODE 1"\n'
y += "DLINE-NODE TOPOLOGY:\n"
y += "".join(f'  - "NODE {i} DLINE 1"\n' for i in left)
y += "".join(f'  - "NODE {i} DLINE 2"\n' for i in top)
y += "".join(f'  - "NODE {i} DLINE 3"\n' for i in bottom)
y += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in coords)
y += elesect + ":\n" + "".join(f'  - "{s}"\n' for s in els)
sys.stdout.write(y)
FLUIDPY
}

NODE=25
value() { grep -oP 'actresult=\s*\K[-0-9.e+]+' "$TMP/$1.log" | head -1; }

fluid "{\"visc\":0.001,\"results\":[[$NODE,\"velx\",0.0,1e-30]]}" "$TMP/default.yaml"
fluid "{\"visc\":0.001,\"stab_body\":\"  STABTYPE: \\\"no_stabilization\\\"\n\",\"results\":[[$NODE,\"velx\",0.0,1e-30]]}" \
      "$TMP/nostab.yaml"
fluid "{\"visc\":0.001,\"stab_body\":\"  DEFINITION_TAU: \\\"Taylor_Hughes_Zarins\\\"\n\",\"results\":[[$NODE,\"velx\",0.0,1e-30]]}" \
      "$TMP/tauok.yaml"
fluid "{\"stab_section\":\"FLUID DYNAMIC/STABILIZATION\",\"stab_body\":\"  SUPG: false\n\"}" "$TMP/stabsec.yaml"
fluid "{\"stab_body\":\"  TAU_TYPE: \\\"Codina\\\"\n\"}" "$TMP/tautype.yaml"
fluid "{\"stab_body\":\"  TAU_DEF: \\\"Codina\\\"\n\"}"  "$TMP/taudef.yaml"
fluid "{\"stab_body\":\"  GRAD-DIV: false\n\"}"          "$TMP/graddiv.yaml"

probe DEFAULT "$TMP/default.yaml"
probe NOSTAB  "$TMP/nostab.yaml"
probe TAUOK   "$TMP/tauok.yaml"
probe STABSEC "$TMP/stabsec.yaml"
probe TAUTYPE "$TMP/tautype.yaml"
probe TAUDEF  "$TMP/taudef.yaml"
probe GRADDIV "$TMP/graddiv.yaml"

echo "DEFAULT_VELX=$(value DEFAULT)"
echo "NOSTAB_VELX=$(value NOSTAB)"
echo "TAUOK_VELX=$(value TAUOK)"
if [ -n "$(value DEFAULT)" ] && [ "$(value DEFAULT)" != "$(value NOSTAB)" ]; then
  echo "STABILIZATION_CHANGES_THE_ANSWER=yes"
else
  echo "STABILIZATION_CHANGES_THE_ANSWER=no"
fi
if [ -n "$(value TAUOK)" ] && [ "$(value TAUOK)" != "$(value DEFAULT)" ]; then
  echo "TAU_DEFINITION_CHANGES_THE_ANSWER=yes"
else
  echo "TAU_DEFINITION_CHANGES_THE_ANSWER=no"
fi

# The section the entry names does not exist.
grep -m1 -F "Section 'FLUID DYNAMIC/STABILIZATION' is not a valid section name." "$TMP/STABSEC.log"
grep -m1 -F "4C_io_input_file.cpp" "$TMP/STABSEC.log"

# Nor do the two key names, nor the hyphenated flag.  4C echoes the leftover.
grep -m1 -F "Could not match this input" "$TMP/TAUTYPE.log"
grep -m1 -F "4C_io_input_spec_builders.cpp" "$TMP/TAUTYPE.log"
echo "TAU_TYPE_UNUSED=$(grep -c 'TAU_TYPE: \"Codina\"' "$TMP/TAUTYPE.log")"
echo "TAU_DEF_UNUSED=$(grep -c 'TAU_DEF: \"Codina\"' "$TMP/TAUDEF.log")"
echo "HYPHENATED_GRAD_DIV_UNUSED=$(grep -c 'GRAD-DIV: false' "$TMP/GRADDIV.log")"
grep -m1 -F "The following data remains unused:" "$TMP/TAUTYPE.log"

# And the Signal cannot be watched: nothing reports kinetic energy.
echo "KINETIC_ENERGY_REPORTED=$(cat "$TMP"/DEFAULT.log "$TMP"/NOSTAB.log | grep -ci 'kinetic energy')"
exit 0
