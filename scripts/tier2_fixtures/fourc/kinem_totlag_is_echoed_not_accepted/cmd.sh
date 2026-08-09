#!/bin/bash
# Tier-2 for fourc::structural_mechanics#3 — KINEM has exactly two legal values
# on this build, and 'nonlinearTotLag' is 4C's OUTPUT spelling, never an input.
#
# The round trip is the whole point and the last arm shows it: write
# 'KINEM nonlinear' plus one unparseable token and 4C dumps the container it
# just built, in which the value reads
#
#     KINEM : nonlinearTotLag
#
# That echo is where the wrong input spelling comes from.  Fed back in it is
# rejected — 'nonlinear' already IS the total-Lagrangian formulation, and there
# is no separate updated-Lagrangian spelling to steer away from either.
#
# One self-contained 3D deck, six arms.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = trailing part of the element line after "MAT 1 "
cat > "$TMP/$2.4C.yaml" <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-09
  MAXITER: 30
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "S"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "t"
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0.0, 10.0, 0.0]
    FUNCT: [0, 1, 0]
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
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 $1"
YAML
}

arm() { deck "$2" "$1"; probe "$1" "$TMP/$1.4C.yaml"; }

arm NONLINEAR "KINEM nonlinear"
arm LINEAR    "KINEM linear"
arm TOTLAG    "KINEM nonlinearTotLag"
arm CAMELCASE "KINEM NonLinear"
arm UPDLAG    "KINEM updated_lagrange"
arm ECHO      "KINEM nonlinear WHATEVER"

# Both legal spellings run.
grep -m1 -F "processor 0 finished normally" "$TMP/NONLINEAR.log"
grep -m1 -F "processor 0 finished normally" "$TMP/LINEAR.log"

# The three illegal ones are refused with the same enum listing.
grep -m1 -F "Could not parse parameter 'KINEM': invalid value 'nonlinearTotLag'. Valid options are: linear|nonlinear" "$TMP/TOTLAG.log"
grep -m1 -F "Could not parse parameter 'KINEM': invalid value 'NonLinear'" "$TMP/CAMELCASE.log"
grep -m1 -F "Could not parse parameter 'KINEM': invalid value 'updated_lagrange'" "$TMP/UPDLAG.log"

# And here is where 'nonlinearTotLag' comes from: it is what the parser stores
# after reading the legal spelling 'nonlinear'.
grep -m1 -F "After parsing, the line still contains 'WHATEVER'." "$TMP/ECHO.log"
python3 - "$TMP/ECHO.log" "$TMP/NONLINEAR.4C.yaml" <<'PY'
import sys
log = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
line = [l for l in log.split("\n") if "Parsed parameters:" in l]
print("ECHO_DUMP_PRESENT=%s" % ("yes" if line else "no"))
print("ECHO_SPELLS_TOTLAG=%s"
      % ("yes" if line and "KINEM : nonlinearTotLag" in line[0] else "no"))
deck = open(sys.argv[2]).read()
print("DECK_WROTE_PLAIN_NONLINEAR=%s"
      % ("yes" if "KINEM nonlinear\"" in deck else "no"))
PY
exit 0
