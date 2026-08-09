#!/bin/bash
# Tier-2 for fourc::thermal#0 — in a STANDALONE PROBLEMTYPE: Thermo run the
# THERMO-prefixed condition sections are valid section names, parse without a
# word of complaint, and are then silently dropped.  Both families are shown,
# each pair differing in exactly one word:
#
#   DESIGN SURF DIRICH  CONDITIONS        -> reaches the thermo field
#   DESIGN SURF THERMO DIRICH CONDITIONS  -> dropped, nothing is prescribed
#   DESIGN SURF NEUMANN CONDITIONS        -> reaches the thermo field
#   DESIGN SURF THERMO NEUMANN CONDITIONS -> dropped
#
# One 3-hex8 bar, transient (OneStepTheta).  The detector is a
# RESULT DESCRIPTION THERMAL entry on the mid-bar node 5: the plain sections
# give a diffusive profile, the prefixed ones give EXACTLY 0.0 and the run
# still exits 0 looking like a success.  There is no diagnostic in either
# prefixed run — that is the whole pitfall — so its absence is asserted too.
. "$(dirname "$0")/../_lib/preamble.sh"

# $1 = the whole conditions block, $2 = expected temperature at node 5
bar() {
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: "OneStepTheta"
  TIMESTEP: 25
  NUMSTEP: 8
  MAXTIME: 200
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 420
      CONDUCT:
        constant: [52]
$1
DSURF-NODE TOPOLOGY:
  - "NODE 13 DSURFACE 1"
  - "NODE 14 DSURFACE 1"
  - "NODE 15 DSURFACE 1"
  - "NODE 16 DSURFACE 1"
  - "NODE 1 DSURFACE 2"
  - "NODE 2 DSURFACE 2"
  - "NODE 3 DSURFACE 2"
  - "NODE 4 DSURFACE 2"
NODE COORDS:
  - "NODE 1 COORD 1.00000000e+01 0.00000000e+00 5.00000000e-01"
  - "NODE 2 COORD 1.00000000e+01 1.00000000e+00 5.00000000e-01"
  - "NODE 3 COORD 1.00000000e+01 1.00000000e+00 -5.00000000e-01"
  - "NODE 4 COORD 1.00000000e+01 0.00000000e+00 -5.00000000e-01"
  - "NODE 5 COORD 6.66666651e+00 0.00000000e+00 5.00000000e-01"
  - "NODE 6 COORD 6.66666651e+00 1.00000000e+00 5.00000000e-01"
  - "NODE 7 COORD 6.66666651e+00 1.00000000e+00 -5.00000000e-01"
  - "NODE 8 COORD 6.66666651e+00 0.00000000e+00 -5.00000000e-01"
  - "NODE 9 COORD 3.33333325e+00 0.00000000e+00 5.00000000e-01"
  - "NODE 10 COORD 3.33333325e+00 1.00000000e+00 5.00000000e-01"
  - "NODE 11 COORD 3.33333325e+00 1.00000000e+00 -5.00000000e-01"
  - "NODE 12 COORD 3.33333325e+00 0.00000000e+00 -5.00000000e-01"
  - "NODE 13 COORD 0.00000000e+00 0.00000000e+00 5.00000000e-01"
  - "NODE 14 COORD 0.00000000e+00 1.00000000e+00 5.00000000e-01"
  - "NODE 15 COORD 0.00000000e+00 1.00000000e+00 -5.00000000e-01"
  - "NODE 16 COORD 0.00000000e+00 0.00000000e+00 -5.00000000e-01"
THERMO ELEMENTS:
  - "1 THERMO HEX8 1 2 3 4 5 6 7 8 MAT 1"
  - "2 THERMO HEX8 5 6 7 8 9 10 11 12 MAT 1"
  - "3 THERMO HEX8 9 10 11 12 13 14 15 16 MAT 1"
RESULT DESCRIPTION:
  - THERMAL:
      DIS: "thermo"
      NODE: 5
      QUANTITY: "temp"
      VALUE: $2
      TOLERANCE: 1e-06
YAML
}

dirich() {  # $1 = "DIRICH" or "THERMO DIRICH"
printf '%s' "DESIGN SURF $1 CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [100.0]
    FUNCT: [0]"
}
neumann() {  # $1 = "NEUMANN" or "THERMO NEUMANN"
# The cold end is held by a plain DIRICH in BOTH arms; only the heat-flux
# section name differs between them.
printf '%s' "DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF $1 CONDITIONS:
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [20.0]
    FUNCT: [0]"
}

# Both arms of each pair expect the answer the PLAIN section produces.
bar "$(dirich  'DIRICH')"          63.3050962998814626 > "$TMP/d_plain.yaml"
bar "$(dirich  'THERMO DIRICH')"   63.3050962998814626 > "$TMP/d_pref.yaml"
bar "$(neumann 'NEUMANN')"          1.08603291977074412 > "$TMP/n_plain.yaml"
bar "$(neumann 'THERMO NEUMANN')"   1.08603291977074412 > "$TMP/n_pref.yaml"

probe DIRICH_PLAIN  "$TMP/d_plain.yaml"
probe DIRICH_PREF   "$TMP/d_pref.yaml"
probe NEUMANN_PLAIN "$TMP/n_plain.yaml"
probe NEUMANN_PREF  "$TMP/n_pref.yaml"

grep -m1 -F "is CORRECT" "$TMP/DIRICH_PLAIN.log"
grep -m1 -F "is CORRECT" "$TMP/NEUMANN_PLAIN.log"
# The prefixed runs leave the mid-bar node at exactly zero.
grep -m1 -F "is WRONG --> actresult= 0.00000000000000000e+00" "$TMP/DIRICH_PREF.log"
grep -m1 -F "is WRONG --> actresult= 0.00000000000000000e+00" "$TMP/NEUMANN_PREF.log"
# ... and 4C says nothing whatsoever about the dropped section.
echo "PREF_SECTION_NAME_REJECTED=$(cat "$TMP/DIRICH_PREF.log" "$TMP/NEUMANN_PREF.log" | grep -c 'is not a valid section name')"
echo "PREF_DROP_DIAGNOSTIC=$(cat "$TMP/DIRICH_PREF.log" "$TMP/NEUMANN_PREF.log" | grep -ciE 'THERMO (DIRICH|NEUMANN).*(ignor|unus|drop|no effect|not applied)')"
# The prefixed decks reach the same point in the code as the plain ones: the
# discretisation is built and the result test runs.  Nothing failed early.
echo "PREF_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/DIRICH_PREF.log")"
exit 0
