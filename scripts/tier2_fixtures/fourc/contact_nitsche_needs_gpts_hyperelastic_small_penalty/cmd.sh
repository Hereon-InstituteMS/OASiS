#!/bin/bash
# Tier-2 for fourc::contact#10 — STRATEGY: "Nitsche" needs three things the
# mortar Penalty strategy does not, and each one has its own failure.
#
# One two-block deck, -0.3 over ten steps, eight arms:
#
#   (control) STRATEGY Penalty, PENALTYPARAM 1e4       -> exit 0, ten steps
#   (a) Nitsche without ALGORITHM                      -> Unrecognized strategy
#       Nitsche + ALGORITHM NTS / LTS / Mortar         -> the SAME message, so
#                                                         GPTS is the only value
#                                                         that gets past this
#   (b) Nitsche + GPTS + MAT_Struct_StVenantKirchhoff  -> evaluate_cauchy_n_dir_
#                                                         and_derivatives not
#                                                         implemented
#   (c) Nitsche + GPTS + MAT_ElastHyper/CoupNeoHooke:
#           PENALTYPARAM 1e4 (the value the mortar Penalty deck is happy with)
#                                                      -> does not converge
#           PENALTYPARAM 1e1                           -> exit 0, ten steps
#
# (c) is the quiet one: nothing says "your penalty is wrong for Nitsche", the
# run just stops converging, because NITSCHE_PENALTY_ADAPTIVE rescales it.
. "$(dirname "$0")/../_lib/preamble.sh"

deck() {  # $1 STRATEGY, $2 ALGORITHM line (may be empty), $3 material block,
          # $4 PENALTYPARAM, $5 out
cat > "$5" <<YAML
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
  PENALTYPARAM: $4
MORTAR COUPLING:
  LM_DUAL_CONSISTENT: "none"
$2SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Contact_Solver"
MATERIALS:
$3
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

STVK='  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1000.0
      NUE: 0.3
      DENS: 1.0'
HYPER='  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 1.0
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 1000.0
      NUE: 0.3'
GPTS='  ALGORITHM: "GPTS"
'

deck Penalty ""    "$STVK"  1.0e4 "$TMP/penalty.yaml"
deck Nitsche ""    "$STVK"  1.0e4 "$TMP/nit_noalg.yaml"
deck Nitsche '  ALGORITHM: "NTS"
'                  "$STVK"  1.0e4 "$TMP/nit_nts.yaml"
deck Nitsche '  ALGORITHM: "LTS"
'                  "$STVK"  1.0e4 "$TMP/nit_lts.yaml"
deck Nitsche '  ALGORITHM: "Mortar"
'                  "$STVK"  1.0e4 "$TMP/nit_mortar.yaml"
deck Nitsche "$GPTS" "$STVK"  1.0e4 "$TMP/nit_gpts_stvk.yaml"
deck Nitsche "$GPTS" "$HYPER" 1.0e4 "$TMP/nit_gpts_big.yaml"
deck Nitsche "$GPTS" "$HYPER" 1.0e1 "$TMP/nit_gpts_small.yaml"

probe PENALTY_CONTROL   "$TMP/penalty.yaml"
probe NITSCHE_NO_ALG    "$TMP/nit_noalg.yaml"
probe NITSCHE_NTS       "$TMP/nit_nts.yaml"
probe NITSCHE_LTS       "$TMP/nit_lts.yaml"
probe NITSCHE_MORTAR    "$TMP/nit_mortar.yaml"
probe NITSCHE_GPTS_STVK "$TMP/nit_gpts_stvk.yaml"
probe NITSCHE_GPTS_PEN_1E4 "$TMP/nit_gpts_big.yaml"
probe NITSCHE_GPTS_PEN_1E1 "$TMP/nit_gpts_small.yaml"

for a in PENALTY_CONTROL NITSCHE_GPTS_PEN_1E4 NITSCHE_GPTS_PEN_1E1; do
  echo "STEPS_$a=$(grep -c 'Finalised step' "$TMP/$a.log")"
done

# The control: the very same PENALTYPARAM is fine for the mortar Penalty
# strategy.
grep -m1 -F "processor 0 finished normally" "$TMP/PENALTY_CONTROL.log"

# (a) GPTS is the only ALGORITHM that gets Nitsche past the strategy factory.
grep -m1 -F 'Unrecognized strategy: "CONTACT::SolvingStrategy::nitsche"' "$TMP/NITSCHE_NO_ALG.log"
grep -m1 -F "4C_contact_strategy_factory.cpp" "$TMP/NITSCHE_NO_ALG.log"
python3 - "$TMP/NITSCHE_NO_ALG.log" "$TMP/NITSCHE_NTS.log" \
          "$TMP/NITSCHE_LTS.log" "$TMP/NITSCHE_MORTAR.log" <<'PY'
import sys
n = sum(1 for p in sys.argv[1:5]
        if 'Unrecognized strategy: "CONTACT::SolvingStrategy::nitsche"'
        in open(p, "rb").read().decode("utf-8", "replace"))
print("ALGORITHMS_REJECTING_NITSCHE=%d" % n)
PY

# (b) With GPTS, a St-Venant-Kirchhoff material is the next wall.
grep -m1 -F "evaluate_cauchy_n_dir_and_derivatives not implemented for material of type m_stvenant" "$TMP/NITSCHE_GPTS_STVK.log"
grep -m1 -F "4C_mat_so3_material.cpp" "$TMP/NITSCHE_GPTS_STVK.log"

# (c) And with both of those right, the penalty value still has to shrink —
# with no message saying so.
grep -m1 -F "The nonlinear solver did not converge!" "$TMP/NITSCHE_GPTS_PEN_1E4.log"
grep -m1 -F "processor 0 finished normally" "$TMP/NITSCHE_GPTS_PEN_1E1.log"
echo "NITSCHE_1E4_MENTIONS_PENALTYPARAM=$(grep -c 'PENALTYPARAM' "$TMP/NITSCHE_GPTS_PEN_1E4.log")"
exit 0
