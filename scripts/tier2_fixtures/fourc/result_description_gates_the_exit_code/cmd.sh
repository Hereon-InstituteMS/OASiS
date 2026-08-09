#!/bin/bash
# Tier-2 for fourc::input_format#17 and #19 — RESULT DESCRIPTION is
# 4C's built-in numerical self-check.
#
# Claim under test (verified by execution 2026-08-03, 4C 2026.2.0-dev):
#   (a) a RESULT DESCRIPTION entry whose VALUE matches the computed
#       answer prints 'is CORRECT' and the process exits 0;
#   (b) the same entry with a wrong VALUE prints 'is WRONG --> ...'
#       plus 'Result check failed with 1 errors out of 1 tests' and
#       the process exits 1 — so an agent can gate correctness on the
#       exit status alone;
#   (c) a TOLERANCE wide enough to swallow the error reports
#       'is CORRECT' for a badly wrong VALUE — the silent branch that
#       makes a loose tolerance worse than no check at all.
#
# The reference value 4.47909266337460053e-03 was itself obtained by
# running (c)-style with TOLERANCE 1e30 and reading abs(diff).
set -u
# Resolve the 4C binary: explicit override first, then the paths this
# repo has been verified against (2026-08-03 verification host runs
# 4C 2026.2.0-dev at /home/alexander/4C/build/4C).
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# stdbuf is mandatory here: 4C writes the result-test verdict to raw
# std::cout and MPI_Abort discards a block-buffered stdout, which is
# itself pitfall input_format#18.
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }

mk() {  # $1 = VALUE, $2 = TOLERANCE, $3 = file
cat > "$3" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 1
  MAXTIME: 0.1
  TOLDISP: 1e-10
  TOLRES: 1e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0, 1, 0]
    FUNCT: [0, 0, 0]
    TYPE: "Live"
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 1"
  - "NODE 8 DSURFACE 1"
  - "NODE 2 DSURFACE 2"
  - "NODE 3 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000
      NUE: 0.3
      DENS: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 3
      QUANTITY: "dispy"
      VALUE: $1
      TOLERANCE: $2
YAML
}

# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology from both arms that carry
# one: arm (b) gets the CORRECT value, so nothing is wrong to report, and arm (c)
# gets a TIGHT tolerance, so its wrong answer is no longer hidden.  "is WRONG -->
# actresult=", "Result check failed with 1 errors out of 1 tests", "EXIT_B=1" and
# "EXIT_C=0" all disappear and the forbidden "EXIT_B=0" appears — red either way.
MUTATE="${T2_MUTATE:-0}"
B_VALUE=999.0
C_TOLERANCE=1.0
if [ "$MUTATE" = "1" ]; then B_VALUE=4.47909266337460053e-03; C_TOLERANCE=1e-10; fi

echo "=== (a) correct VALUE, tight TOLERANCE — must exit 0 ==="
mk 4.47909266337460053e-03 1e-10 "$TMP/ok.4C.yaml"
run4c "$TMP/ok.4C.yaml" "$TMP/o_ok" | grep -E "is CORRECT|is WRONG" | head -2
echo "EXIT_A=${PIPESTATUS[0]}"

echo "=== (b) wrong VALUE, tight TOLERANCE — must exit 1 ==="
mk "$B_VALUE" 1e-10 "$TMP/bad.4C.yaml"
run4c "$TMP/bad.4C.yaml" "$TMP/o_bad" | grep -E "is WRONG|Result check failed" | head -2
echo "EXIT_B=${PIPESTATUS[0]}"

echo "=== (c) wrong VALUE, loose TOLERANCE — silently exits 0 ==="
mk 0.0 "$C_TOLERANCE" "$TMP/loose.4C.yaml"
run4c "$TMP/loose.4C.yaml" "$TMP/o_loose" | grep -E "is CORRECT|is WRONG" | head -2
echo "EXIT_C=${PIPESTATUS[0]}"
exit 0
