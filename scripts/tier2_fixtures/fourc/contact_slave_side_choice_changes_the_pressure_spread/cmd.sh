#!/bin/bash
# Tier-2 for fourc::contact#4 — with non-matching meshes it matters which side
# is Slave, and the effect is on the load distribution, not on convergence.
#
# The deck deliberately mismatches the two contact surfaces: the lower block is
# split 2x2 in plane (nine nodes on its top face), the upper block is a single
# element (four nodes on its bottom face). Everything else is identical between
# the two arms; only the Side qualifiers are exchanged.
#
#   Slave = the FINE face   -> nine slave nodes carry the interface
#   Slave = the COARSE face -> four
#
# The measurement is the spread of the fine face's own vertical displacement
# between its corner (node 10) and its centre (node 14). With the fine side as
# slave the two agree closely; with the coarse side as slave the mortar
# projection concentrates the transfer at the coarse nodes and the spread grows
# several-fold. Both numbers come out of 4C's result-test report, forced to print
# by an unmatchable RESULT DESCRIPTION.
#
# The half of the entry that does NOT hold up: "swapping slave and master can
# cause convergence issues". Both arms take exactly the same number of Newton
# iterations here, which is what SWAP_CHANGES_ITERATION_COUNT records.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 = Side of the fine face (E2), $2 = Side of the coarse face (E3)
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
CONTACT DYNAMIC:
  LINEAR_SOLVER: 2
  STRATEGY: "Penalty"
  PENALTYPARAM: 1.0e4
MORTAR COUPLING:
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
    Side: "$1"
  - E: 3
    InterfaceID: 1
    Side: "$2"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 10
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e-30
  - STRUCTURE:
      DIS: "structure"
      NODE: 14
      QUANTITY: "dispz"
      VALUE: 0.0
      TOLERANCE: 1.0e-30
DSURF-NODE TOPOLOGY:
  - "NODE 1 DSURFACE 1"
  - "NODE 2 DSURFACE 1"
  - "NODE 3 DSURFACE 1"
  - "NODE 4 DSURFACE 1"
  - "NODE 5 DSURFACE 1"
  - "NODE 6 DSURFACE 1"
  - "NODE 7 DSURFACE 1"
  - "NODE 8 DSURFACE 1"
  - "NODE 9 DSURFACE 1"
  - "NODE 10 DSURFACE 2"
  - "NODE 11 DSURFACE 2"
  - "NODE 12 DSURFACE 2"
  - "NODE 13 DSURFACE 2"
  - "NODE 14 DSURFACE 2"
  - "NODE 15 DSURFACE 2"
  - "NODE 16 DSURFACE 2"
  - "NODE 17 DSURFACE 2"
  - "NODE 18 DSURFACE 2"
  - "NODE 19 DSURFACE 3"
  - "NODE 20 DSURFACE 3"
  - "NODE 21 DSURFACE 3"
  - "NODE 22 DSURFACE 3"
  - "NODE 23 DSURFACE 4"
  - "NODE 24 DSURFACE 4"
  - "NODE 25 DSURFACE 4"
  - "NODE 26 DSURFACE 4"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 0.5 0.0 0.0"
  - "NODE 3 COORD 1.0 0.0 0.0"
  - "NODE 4 COORD 0.0 0.5 0.0"
  - "NODE 5 COORD 0.5 0.5 0.0"
  - "NODE 6 COORD 1.0 0.5 0.0"
  - "NODE 7 COORD 0.0 1.0 0.0"
  - "NODE 8 COORD 0.5 1.0 0.0"
  - "NODE 9 COORD 1.0 1.0 0.0"
  - "NODE 10 COORD 0.0 0.0 1.0"
  - "NODE 11 COORD 0.5 0.0 1.0"
  - "NODE 12 COORD 1.0 0.0 1.0"
  - "NODE 13 COORD 0.0 0.5 1.0"
  - "NODE 14 COORD 0.5 0.5 1.0"
  - "NODE 15 COORD 1.0 0.5 1.0"
  - "NODE 16 COORD 0.0 1.0 1.0"
  - "NODE 17 COORD 0.5 1.0 1.0"
  - "NODE 18 COORD 1.0 1.0 1.0"
  - "NODE 19 COORD 0.0 0.0 1.1"
  - "NODE 20 COORD 1.0 0.0 1.1"
  - "NODE 21 COORD 1.0 1.0 1.1"
  - "NODE 22 COORD 0.0 1.0 1.1"
  - "NODE 23 COORD 0.0 0.0 2.1"
  - "NODE 24 COORD 1.0 0.0 2.1"
  - "NODE 25 COORD 1.0 1.0 2.1"
  - "NODE 26 COORD 0.0 1.0 2.1"
STRUCTURE ELEMENTS:
  - "1 SOLID HEX8 1 2 5 4 10 11 14 13 MAT 1 KINEM nonlinear"
  - "2 SOLID HEX8 2 3 6 5 11 12 15 14 MAT 1 KINEM nonlinear"
  - "3 SOLID HEX8 4 5 8 7 13 14 17 16 MAT 1 KINEM nonlinear"
  - "4 SOLID HEX8 5 6 9 8 14 15 18 17 MAT 1 KINEM nonlinear"
  - "5 SOLID HEX8 19 20 21 22 23 24 25 26 MAT 1 KINEM nonlinear"
YAML
}

deck "Slave"  "Master" > "$TMP/fine_slave.yaml"
deck "Master" "Slave"  > "$TMP/coarse_slave.yaml"

probe FINESLAVE   "$TMP/fine_slave.yaml"
probe COARSESLAVE "$TMP/coarse_slave.yaml"

# Both arms complete; the active set size is just the slave node count.
echo "FINESLAVE_STEPS=$(grep -c 'Finalised step' "$TMP/FINESLAVE.log")"
echo "COARSESLAVE_STEPS=$(grep -c 'Finalised step' "$TMP/COARSESLAVE.log")"
grep -m1 -F "Contact-Normal-Active-Set-Size = 9" "$TMP/FINESLAVE.log"
grep -m1 -F "Contact-Normal-Active-Set-Size = 4" "$TMP/COARSESLAVE.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/FINESLAVE.log"

python3 - "$TMP/FINESLAVE.log" "$TMP/COARSESLAVE.log" <<'PY'
import re, sys
def read(p):
    t = open(p, "rb").read().decode("utf-8", "replace")
    v = {}
    for m in re.finditer(r'dispz +at node +(\d+)\s+is WRONG --> actresult=\s*([-0-9.e+]+),', t):
        v[m.group(1)] = float(m.group(2))
    iters = sum(int(m) for m in re.findall(r'nlniter (\d+)', t))
    return v, iters

a, ia = read(sys.argv[1])     # fine face is Slave
b, ib = read(sys.argv[2])     # coarse face is Slave
assert {"10", "14"} <= set(a) and {"10", "14"} <= set(b), "4C did not report both fine-face nodes"
spread_fine   = abs(a["10"] - a["14"])
spread_coarse = abs(b["10"] - b["14"])
print("FINE_FACE_SPREAD_WITH_FINE_SLAVE=%.6e" % spread_fine)
print("FINE_FACE_SPREAD_WITH_COARSE_SLAVE=%.6e" % spread_coarse)
print("COARSE_SLAVE_SPREADS_MORE=%s" % ("yes" if spread_coarse > spread_fine else "no"))
print("COARSE_SLAVE_SPREADS_AT_LEAST_3X=%s"
      % ("yes" if spread_coarse > 3.0 * spread_fine else "no"))
print("SWAP_CHANGES_ITERATION_COUNT=%s" % ("yes" if ia != ib else "no"))
PY
exit 0
