#!/bin/bash
# Tier-2 for fourc::input_format#18 — the RESULT DESCRIPTION failure
# diagnostic survives ONLY under line-buffered stdout.
#
# Claim under test (corrected 2026-08-03 by adversarial re-audit; the
# original text said "through a line-buffered pipe (stdbuf -oL -eL 4C
# in.4C.yaml out) or to a file", and the "or to a file" half is FALSE):
#
#   (a) plain pipe            -> 'is WRONG' LOST, output ends at
#                                'Checking results of N tests:'
#   (b) plain file redirect   -> 'is WRONG' LOST as well (a regular
#                                file is fully buffered, not line
#                                buffered — this is the half that the
#                                original claim got wrong)
#   (c) stdbuf -oL -eL + file -> 'is WRONG' AND 'Result check failed'
#                                both present
#
# The deck is the same single-HEX8 cantilever the
# result_description_gates_the_exit_code fixture uses; its true dispy at
# node 3 is 4.47909266337460053e-03, so VALUE 999.0 always fails.
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/bad.4C.yaml" <<'YAML'
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
      VALUE: 999.0
      TOLERANCE: 1e-10
YAML

IN="$TMP/bad.4C.yaml"

# (a) plain pipe, stdout only.  The pipeline must run in THIS shell, not
# in a $(...) subshell, or PIPESTATUS[0] reports the subshell instead of
# 4C and the exit code silently reads 0.
"$BIN" "$IN" "$TMP/oa" 2>/dev/null | tail -1 > "$TMP/pipetail.txt"
PIPE_RC=${PIPESTATUS[0]}
PIPE_TAIL=$(cat "$TMP/pipetail.txt")
case "$PIPE_TAIL" in
  *"Checking results of 1 tests:"*) PIPE_ENDS=yes ;;
  *) PIPE_ENDS=no ;;
esac
echo "plain_pipe: EXIT=$PIPE_RC ENDS_AT_CHECKING=$PIPE_ENDS"

# (b) plain redirect to a FILE — the half the original claim got wrong.
# Repeat three times so a one-off scheduling fluke cannot pass it.
FILE_HITS=0
for i in 1 2 3; do
  "$BIN" "$IN" "$TMP/ob$i" > "$TMP/f$i.log" 2>&1
  n=$(grep -c "is WRONG" "$TMP/f$i.log" || true)
  FILE_HITS=$((FILE_HITS + n))
done
echo "plain_file_redirect: IS_WRONG_LINES_IN_3_RUNS=$FILE_HITS"

# (c) stdbuf + the same file redirect
stdbuf -oL -eL "$BIN" "$IN" "$TMP/oc" > "$TMP/g.log" 2>&1
G_RC=$?
G_WRONG=$(grep -c "is WRONG" "$TMP/g.log" || true)
G_FAIL=$(grep -c "Result check failed" "$TMP/g.log" || true)
echo "stdbuf_file_redirect: EXIT=$G_RC IS_WRONG=$G_WRONG RESULT_CHECK_FAILED=$G_FAIL"

exit 0
