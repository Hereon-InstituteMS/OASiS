#!/bin/bash
# Tier-2 for fourc::input_format#7 — the out-of-plane depth of a 2D structural
# element is not a cosmetic number.  It multiplies the element stiffness while
# a Live line load is NOT scaled with it, so putting the element edge length
# (or any other geometric width) there rescales the whole answer by exactly
# that factor, with no warning and exit 0.
#
# ONE CORRECTION TO THE WORDING.  On this build the keyword is THICK on WALL.
# There is no THICKNESS key on any legacy element and no PLANE_ASSUMPTION key
# anywhere in the schema, so a deck written with the newer spelling does not
# merely behave differently -- it does not parse.  Both facts are read out of
# `4C --parameters` here so the correction stays pinned.
. "$(dirname "$0")/../_lib/preamble.sh"

mk() {  # $1 = THICK value, $2 = out file
cat > "$2" <<YAML
PROBLEM SIZE:
  DIM: 2
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
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 2
    ONOFF: [0, 1]
    VAL: [0, 1]
    FUNCT: [0, 0]
    TYPE: "Live"
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
  - "NODE 2 DLINE 2"
  - "NODE 3 DLINE 2"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM linear EAS none THICK $1 STRESS_STRAIN plane_strain GP 2 2"
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
      VALUE: 0.0
      TOLERANCE: 1e9
YAML
}

# Same physical problem, four different out-of-plane depths.
mk 0.5 "$TMP/t05.yaml"
mk 1.0 "$TMP/t10.yaml"
mk 2.0 "$TMP/t20.yaml"
mk 4.0 "$TMP/t40.yaml"

probe T05 "$TMP/t05.yaml"
probe T10 "$TMP/t10.yaml"
probe T20 "$TMP/t20.yaml"
probe T40 "$TMP/t40.yaml"

# Every arm converges and exits 0; nothing in the log hints at a problem.
grep -m1 -F "processor 0 finished normally" "$TMP/T10.log"
echo "ANY_ARM_WARNED_ABOUT_THICKNESS=$(cat "$TMP"/T*.log | grep -ci 'thick')"
val() { grep -m1 -o 'abs(diff)= [0-9.e+-]*' "$1" | awk '{print $2}'; }
echo "DISPY_THICK_0p5=$(val "$TMP/T05.log")"
echo "DISPY_THICK_1p0=$(val "$TMP/T10.log")"
echo "DISPY_THICK_2p0=$(val "$TMP/T20.log")"
echo "DISPY_THICK_4p0=$(val "$TMP/T40.log")"
# The product dispy * THICK is invariant: the depth divides the answer exactly.
python3 - "$(val "$TMP/T05.log")" "$(val "$TMP/T10.log")" "$(val "$TMP/T20.log")" "$(val "$TMP/T40.log")" <<'PY'
import sys
d05, d10, d20, d40 = (float(x) for x in sys.argv[1:5])
prods = [d05*0.5, d10*1.0, d20*2.0, d40*4.0]
ok = all(abs(p - prods[0]) <= 1e-12*abs(prods[0]) for p in prods)
print("DISPY_TIMES_THICK_IS_INVARIANT=%s" % ("yes" if ok else "no"))
print("RATIO_THICK1_OVER_THICK2=%.6f" % (d10/d20))
print("RATIO_THICK1_OVER_THICK4=%.6f" % (d10/d40))
PY

# The keyword names, read off the binary rather than assumed.
"$BIN" --parameters 2>/dev/null > "$TMP/params.yaml"
ele=$(awk '/^legacy_element_specs:$/{f=1;next} /^[A-Za-z_$]/{f=0} f' "$TMP/params.yaml")
echo "SCHEMA_ELEMENT_KEY_THICK=$(printf '%s\n' "$ele" | grep -c 'name: THICK$')"
echo "SCHEMA_ELEMENT_KEY_THICKNESS=$(printf '%s\n' "$ele" | grep -c 'name: THICKNESS$')"
echo "SCHEMA_HAS_PLANE_ASSUMPTION_ANYWHERE=$(grep -c 'PLANE_ASSUMPTION' "$TMP/params.yaml")"
exit 0
