#!/bin/bash
# Tier-2 for fourc::contact#11 — appearing in the enum that `4C --parameters`
# prints is NECESSARY, not sufficient.
#
# Two settings pass the schema and are then refused by the code:
#
#   STRATEGY: "Uzawa"           -> This contact strategy is not yet considered!
#   SEMI_SMOOTH_NEWTON: false   -> Currently we support only the semi-smooth
#                                  Newton case!
#
# The fixture reads the schema out of the binary itself, so the "it IS listed"
# half is evidence and not an assertion: UZAWA_IS_IN_THE_STRATEGY_ENUM and
# SEMI_SMOOTH_NEWTON_IS_A_DOCUMENTED_KEY come from `4C --parameters`.
#
# A genuinely unknown STRATEGY is rejected earlier and differently, which is how
# you can tell the schema really is filtering: 'Could not match this input'
# against 'This contact strategy is not yet considered!'.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 STRATEGY, $2 extra CONTACT DYNAMIC lines, $3 out
cat > "$3" <<YAML
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
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "$1"
  PENALTYPARAM: 1.0e4
$2MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
SOLVER 1:
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
DESIGN SURF MORTAR CONTACT CONDITIONS 3D:
  - E: 2
    InterfaceID: 1
    Side: "Master"
  - E: 3
    InterfaceID: 1
    Side: "Slave"
DSURF-NODE TOPOLOGY:
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

deck "Penalty" ""                              "$TMP/control.yaml"
deck "Uzawa"   ""                              "$TMP/uzawa.yaml"
deck "Penalty" '  SEMI_SMOOTH_NEWTON: false
'                                              "$TMP/nosemi.yaml"
deck "Steamroller" ""                          "$TMP/unknown.yaml"

probe CONTROL      "$TMP/control.yaml"
probe UZAWA        "$TMP/uzawa.yaml"
probe NO_SEMISMOOTH "$TMP/nosemi.yaml"
probe UNKNOWN      "$TMP/unknown.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/CONTROL.log"

# Listed by the schema, refused by the code.
grep -m1 -F "This contact strategy is not yet considered!" "$TMP/UZAWA.log"
grep -m1 -F "4C_contact_strategy_factory.cpp" "$TMP/UZAWA.log"
grep -m1 -F "Currently we support only the semi-smooth Newton case!" "$TMP/NO_SEMISMOOTH.log"
grep -m1 -F "4C_contact_noxinterface.cpp" "$TMP/NO_SEMISMOOTH.log"

# A value that is NOT in the schema fails earlier and differently.
grep -m1 -F "Could not match this input" "$TMP/UNKNOWN.log"
echo "UNKNOWN_STRATEGY_REACHED_THE_FACTORY=$(grep -c 'not yet considered' "$TMP/UNKNOWN.log")"

# ...and the schema really does advertise both settings.  Read it out of the
# binary rather than asserting it.
"$BIN" --parameters > "$TMP/params.txt" 2>/dev/null
python3 - "$TMP/params.txt" <<'PY'
import sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
i = t.index("- name: CONTACT DYNAMIC\n")
seg = t[i:i + 12000]
j = seg.index("- name: STRATEGY")
strategy = seg[j:j + 900]
print("UZAWA_IS_IN_THE_STRATEGY_ENUM=%s"
      % ("yes" if '- name: "Uzawa"' in strategy else "no"))
print("SEMI_SMOOTH_NEWTON_IS_A_DOCUMENTED_KEY=%s"
      % ("yes" if "- name: SEMI_SMOOTH_NEWTON" in seg else "no"))
print("STEAMROLLER_IS_IN_THE_STRATEGY_ENUM=%s"
      % ("yes" if "Steamroller" in strategy else "no"))
PY
exit 0
