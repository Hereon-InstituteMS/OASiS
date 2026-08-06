#!/bin/bash
# Tier-2 for fourc::thermal#4 — MAT_scatra in a PROBLEMTYPE: Thermo deck is not
# diagnosed.  The process dies with SIGSEGV and shell status 139.  Two decks
# identical except for the material block:
#
#   MAT_Fourier {CAPA, CONDUCT}  -> runs, result test CORRECT, exit 0
#   MAT_scatra  {DIFFUSIVITY}    -> exit 139
#
# The claim said there is "NO 4C output at all - no message, just a shell exit
# status of 139".  That half is wrong and worth knowing, because it tells you
# WHERE the crash is: 4C parses the deck, builds the discretisation, prints the
# `fill_complete() on discretization thermo` banner and the "Welcome to Thermal
# Time Integration" banner, and only then dereferences a null material inside
# the Thermo::TimIntImpl constructor.  What is genuinely absent is the
# `PROC 0 ERROR` block — the crash is not a FOUR_C_THROW — but OpenMPI's signal
# handler does print a backtrace naming the constructor.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = the material body
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [100.0]
    FUNCT: [0]
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
THERMO ELEMENTS:
  - "1 THERMO HEX8 1 2 3 4 5 6 7 8 MAT 1"
MATERIALS:
  - MAT: 1
$1
RESULT DESCRIPTION:
  - THERMAL:
      DIS: "thermo"
      NODE: 2
      QUANTITY: "temp"
      VALUE: 100.0
      TOLERANCE: 1e-8
YAML
}

FOURIER='    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [1.0]'
SCATRA='    MAT_scatra:
      DIFFUSIVITY: 1.0'

deck "$FOURIER" > "$TMP/fourier.yaml"
deck "$SCATRA"  > "$TMP/scatra.yaml"

probe FOURIER "$TMP/fourier.yaml"
# Three repeats, because "it segfaults" is only worth writing down if it is
# deterministic.
probe SCATRA1 "$TMP/scatra.yaml"
probe SCATRA2 "$TMP/scatra.yaml"
probe SCATRA3 "$TMP/scatra.yaml"

grep -m1 -F "is CORRECT" "$TMP/FOURIER.log"
grep -m1 -F "Signal: Segmentation fault (11)" "$TMP/SCATRA1.log"
# It is NOT a FOUR_C_THROW: no PROC 0 ERROR block, no source location, no text.
echo "SCATRA_FOUR_C_ERROR_BLOCKS=$(grep -c 'PROC 0 ERROR' "$TMP/SCATRA1.log")"
# But it is not silent either: 4C gets all the way through input and setup.
echo "SCATRA_REACHED_FILL_COMPLETE=$(grep -c 'fill_complete() on discretization thermo' "$TMP/SCATRA1.log")"
grep -m1 -F "Welcome to Thermal Time Integration" "$TMP/SCATRA1.log"
# The OpenMPI backtrace names the frame that dereferenced the null material.
echo "SCATRA_BACKTRACE_NAMES_TIMINTIMPL=$(grep -c 'Thermo10TimIntImpl' "$TMP/SCATRA1.log")"
exit 0
