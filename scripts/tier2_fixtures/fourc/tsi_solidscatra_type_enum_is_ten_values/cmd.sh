#!/bin/bash
# Tier-2 for fourc::tsi#11 — the two lists in this claim, both swept by
# execution.  Both are wrong.
#
# TYPE ENUM.  The claim says 11 values including 'NLS'.  Running upstream
# tsi_lincompression_1waydisp once per value shows 10 accepted and NLS refused
# with the same message an invented value gets:
#     The input type NLS is not valid for SOLIDSCATRA elements!
# The origin file is 4C_solid_scatra_3D_ele_lib.cpp, not the
# 4C_solid_scatra_ele_lib.cpp the claim names.
#
# CELL TYPES.  The claim says SOLIDSCATRA "also supports QUAD4, QUAD9, TRI3,
# TRI6, HEX27, TET4, TET10, NURBS27 — not just HEX8".  The 3D half is right and
# the 2D half is not: every 2D cell type is refused by the element definition
# table with
#     Element 'SOLIDSCATRA' does not seem to know cell type 'quad4'.
# and so are pyramid5 and wedge6.  What the element declares is exactly
# {hex8, hex27, tet4, tet10, nurbs27}.  An unsupported cell type is detected by
# that message; a supported one gets past it and fails later on this deliberately
# degenerate probe geometry, which is the discriminator used below.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream tsi_lincompression_1waydisp.4C.yaml) || exit 3
grep -q "KINEM linear TYPE Undefined" "$BASE" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

# ---- TYPE enum sweep ----------------------------------------------------
CLAIMED_TYPES="Undefined AdvReac CardMono GR NLS Chemo ChemoReac ElchDiffCond \
ElchElectrode Loma Std"
ok=0; bad=0; badnames=""
for v in $CLAIMED_TYPES; do
  sed "s/KINEM linear TYPE Undefined/KINEM linear TYPE $v/" "$BASE" > "$TMP/t.yaml"
  stdbuf -oL -eL "$BIN" "$TMP/t.yaml" "$TMP/o_t" > "$TMP/t_$v.log" 2>&1
  if grep -q "not valid for SOLIDSCATRA elements" "$TMP/t_$v.log"; then
    bad=$((bad + 1)); badnames="$badnames $v"
  else
    ok=$((ok + 1))
  fi
done
echo "CLAIMED_TYPE_VALUES=11"
echo "TYPE_VALUES_ACCEPTED=$ok"
echo "TYPE_VALUES_REFUSED=$bad"
echo "TYPE_VALUES_REFUSED_NAMES=$(echo $badnames)"
grep -m1 -F "The input type NLS is not valid for SOLIDSCATRA elements!" "$TMP/t_NLS.log"
grep -m1 -oF "4C_solid_scatra_3D_ele_lib.cpp" "$TMP/t_NLS.log"

# ---- cell-type sweep ----------------------------------------------------
celldeck() {  # $1 = cell type token, $2 = node ids
cat <<YAML
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  MAXTIME: 0.1
  LINEAR_SOLVER: 2
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 0.1
  MAXTIME: 0.1
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 0.1
  TIMESTEP: 0.1
  ITEMAX: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [1e+11]
      NUE: 0
      DENS: 1
      THEXPANS: 1e-05
      INITTEMP: 293
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: 420
      CONDUCT:
        constant: [52]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "393.0"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0, 0]
    FUNCT: [0, 0]
DLINE-NODE TOPOLOGY:
  - "NODE 1 DLINE 1"
  - "NODE 4 DLINE 1"
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "1 SOLIDSCATRA $1 $2 MAT 1 KINEM linear TYPE Undefined"
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: 2
      QUANTITY: "dispx"
      VALUE: 0.001
      TOLERANCE: 1e-09
YAML
}

nodes() { python3 -c "print(' '.join(['1'] * $1))"; }
unsupported=""; supported=""
for pair in QUAD4:4 QUAD9:9 TRI3:3 TRI6:6 HEX8:8 HEX27:27 TET4:4 TET10:10 \
            NURBS27:27 PYRAMID5:5 WEDGE6:6; do
  ct="${pair%%:*}"; n="${pair##*:}"
  celldeck "$ct" "$(nodes "$n")" > "$TMP/c.yaml"
  stdbuf -oL -eL "$BIN" "$TMP/c.yaml" "$TMP/o_c" > "$TMP/c_$ct.log" 2>&1
  if grep -q "does not seem to know cell type" "$TMP/c_$ct.log"; then
    unsupported="$unsupported $ct"
  else
    supported="$supported $ct"
  fi
done
echo "CELLTYPES_KNOWN_TO_SOLIDSCATRA=$(echo $supported)"
echo "CELLTYPES_REFUSED_BY_SOLIDSCATRA=$(echo $unsupported)"
grep -m1 -F "Element 'SOLIDSCATRA' does not seem to know cell type 'quad4'." "$TMP/c_QUAD4.log"
grep -m1 -F "Element 'SOLIDSCATRA' does not seem to know cell type 'tri6'." "$TMP/c_TRI6.log"
grep -m1 -oF "4C_fem_general_element_definition.cpp" "$TMP/c_QUAD4.log"
# The catalogued origin file for the TYPE message does not exist in the output.
echo "CLAIMED_LIB_FILE_WITHOUT_3D=$(grep -c 'src/solid_scatra_ele/4C_solid_scatra_ele_lib.cpp' "$TMP/t_NLS.log")"
exit 0
