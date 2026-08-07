#!/bin/bash
# Tier-2 for fourc::contact — mortar contact adds exactly THREE sections to a
# structural deck, and each way of getting them wrong has its own diagnostic.
# This fixture builds ONE working two-block penalty-contact deck and then
# breaks it in five different ways, asserting the message each time.
#
# The deck is self-contained: inline nodes, inline elements, no mesh file.
#
# Sixth arm is the dangerous one: deleting the contact CONDITION section
# while leaving CONTACT DYNAMIC and MORTAR COUPLING in place runs to
# completion with exit 0 and never mentions contact at all.
set -u
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# stdbuf is mandatory: 4C aborts via MPI_Abort and a block-buffered stdout
# (including a plain file redirect) discards the diagnostic.
run4c() { stdbuf -oL -eL "$BIN" "$1" "$2" 2>&1; }

deck() {  # $1 = contact-dynamic block, $2 = mortar block, $3 = condition block
cat <<YAML
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1.0
  TOLDISP: 1.0e-08
  TOLRES: 1.0e-06
  MAXITER: 50
  LINEAR_SOLVER: 1
$1$2SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
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
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, -0.3]
    FUNCT: [0, 0, 1]
$3DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 2"
  - "NODE 6 DSURFACE 2"
  - "NODE 7 DSURFACE 2"
  - "NODE 8 DSURFACE 2"
  - "NODE 9 DSURFACE 3"
  - "NODE 10 DSURFACE 3"
  - "NODE 11 DSURFACE 3"
  - "NODE 12 DSURFACE 3"
  - "NODE 13 DSURFACE 4"
  - "NODE 14 DSURFACE 4"
  - "NODE 15 DSURFACE 4"
  - "NODE 16 DSURFACE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
  - "NODE 5 COORD 0.0 0.0 1.0"
  - "NODE 6 COORD 1.0 0.0 1.0"
  - "NODE 7 COORD 1.0 1.0 1.0"
  - "NODE 8 COORD 0.0 1.0 1.0"
  - "NODE 9 COORD 0.0 0.0 1.1"
  - "NODE 10 COORD 1.0 0.0 1.1"
  - "NODE 11 COORD 1.0 1.0 1.1"
  - "NODE 12 COORD 0.0 1.0 1.1"
  - "NODE 13 COORD 0.0 0.0 2.1"
  - "NODE 14 COORD 1.0 0.0 2.1"
  - "NODE 15 COORD 1.0 1.0 2.1"
  - "NODE 16 COORD 0.0 1.0 2.1"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear"
  - "2 SOLID HEX8 9 10 11 12 13 14 15 16 MAT 1 KINEM nonlinear"
YAML
}

CD_OK='CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
'
CD_NOSOLVER='CONTACT DYNAMIC:
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
'
CD_NOPEN='CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
'
MO_OK='MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
'
MO_DEFAULT='MORTAR COUPLING:
  LM_SHAPEFCN: "Dual"
'
CO_OK='DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
'
CO_TWOMASTER='DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Master"
'
CO_MISMATCH='DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 2
    Side: "Slave"
'

arm() {  # $1 label  $2 cd  $3 mortar  $4 cond
  deck "$2" "$3" "$4" > "$TMP/$1.4C.yaml"
  run4c "$TMP/$1.4C.yaml" "$TMP/o_$1" > "$TMP/$1.log" 2>&1
  echo "EXIT_$1=$?"
}

# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology from all seven broken
# arms: each is rebuilt from the three CORRECT blocks the GOOD arm uses.  Every
# EXIT_*=1 expectation becomes 0, all five quoted 4C diagnostics disappear, and
# the NOCONDITION arm — the dangerous silent one — now carries its contact
# condition, so it builds the interface and the forbidden
# NOCONDITION_BUILDS_INTERFACE=1 appears.  The fixture must go red.
MUTATE="${T2_MUTATE:-0}"
CD_NOSOLVER_A="$CD_NOSOLVER"; CD_NONE_A=""; CD_NOPEN_A="$CD_NOPEN"
MO_DEFAULT_A="$MO_DEFAULT"; CO_TWOMASTER_A="$CO_TWOMASTER"
CO_MISMATCH_A="$CO_MISMATCH"; CO_NONE_A=""
if [ "$MUTATE" = "1" ]; then
  CD_NOSOLVER_A="$CD_OK"; CD_NONE_A="$CD_OK"; CD_NOPEN_A="$CD_OK"
  MO_DEFAULT_A="$MO_OK"; CO_TWOMASTER_A="$CO_OK"
  CO_MISMATCH_A="$CO_OK"; CO_NONE_A="$CO_OK"
fi

arm GOOD        "$CD_OK"          "$MO_OK"          "$CO_OK"
arm NOSOLVER    "$CD_NOSOLVER_A"  "$MO_OK"          "$CO_OK"
arm NOCONTACTSEC "$CD_NONE_A"     "$MO_OK"          "$CO_OK"
arm NOPENALTY   "$CD_NOPEN_A"     "$MO_OK"          "$CO_OK"
arm MORTARDEF   "$CD_OK"          "$MO_DEFAULT_A"   "$CO_OK"
arm TWOMASTER   "$CD_OK"          "$MO_OK"          "$CO_TWOMASTER_A"
arm MISMATCH    "$CD_OK"          "$MO_OK"          "$CO_MISMATCH_A"
arm NOCONDITION "$CD_OK"          "$MO_OK"          "$CO_NONE_A"

echo "GOOD_STEPS=$(grep -c 'Finalised step' "$TMP/GOOD.log")"
echo "GOOD_BUILDS_INTERFACE=$(grep -c 'Building contact interface' "$TMP/GOOD.log")"
# The silent arm: no contact anywhere in the output, yet exit 0.
echo "NOCONDITION_STEPS=$(grep -c 'Finalised step' "$TMP/NOCONDITION.log")"
echo "NOCONDITION_BUILDS_INTERFACE=$(grep -c 'Building contact interface' "$TMP/NOCONDITION.log")"

for a in NOSOLVER NOCONTACTSEC NOPENALTY MORTARDEF TWOMASTER MISMATCH; do
  echo "--- $a ---"
  grep -m1 -h -F -e 'no linear solver defined for meshtying/contact problem' \
      -e 'Penalty parameter eps = 0, must be greater than 0' \
      -e 'Consistent dual shape functions in boundary elements only for Lagrange multiplier strategy.' \
      -e 'Slave side missing in contact condition group!' \
      -e 'Cannot find matching contact condition for id' "$TMP/$a.log"
done
exit 0
