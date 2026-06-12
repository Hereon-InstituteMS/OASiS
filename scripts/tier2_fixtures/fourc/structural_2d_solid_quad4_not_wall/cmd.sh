#!/bin/bash
# Tier-2: 4C 2D structural element type is SOLID QUAD4, NOT WALL.
#
# Multiple catalog passages talk about "WALL TRI3", "WALL QUAD4"
# etc. for 2D 4C structural simulations (including a contact
# template the prepare_simulation tool serves). In 4C 2026.3 the
# string "WALL" itself is unknown:
#   PROC 0 ERROR in 4C_comm_parobjectfactory.cpp:153:
#   Unknown type 'WALL' of finite element
# Catalog further used 'THICK 1.0' and 'STRESS_STRAIN plane_strain'
# — the real syntax uses 'THICKNESS' and 'PLANE_ASSUMPTION'.
#
# Verified against tests/input_files/contact2D_initfield.4C.yaml
# (4C source tree) — the canonical 2D structural element is
# "SOLID QUAD4 ... MAT 1 KINEM nonlinear THICKNESS 1.0
#  PLANE_ASSUMPTION plane_strain".
set -u
BIN=$HOME/Schreibtisch/4C-src/4C/build/4C
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/wall.yaml" <<'EOF'
PROBLEM TYPE:
  PROBLEMTYPE: Structure
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: Statics
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1.0
      NUE: 0.3
      DENS: 1.0
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "1 WALL QUAD4 1 2 3 4 MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain"
EOF
echo "=== WALL (catalog form) — must fail ==="
"$BIN" "$TMP/wall.yaml" "$TMP/out_wall" 2>&1 | head -25
echo
cat > "$TMP/solid.yaml" <<'EOF'
PROBLEM TYPE:
  PROBLEMTYPE: Structure
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: Statics
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1.0
      NUE: 0.3
      DENS: 1.0
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
  - "NODE 3 COORD 1.0 1.0 0.0"
  - "NODE 4 COORD 0.0 1.0 0.0"
STRUCTURE ELEMENTS:
  - "1 SOLID QUAD4 1 2 3 4 MAT 1 KINEM nonlinear THICKNESS 1.0 PLANE_ASSUMPTION plane_strain"
EOF
echo "=== SOLID QUAD4 + THICKNESS + PLANE_ASSUMPTION — must reach fill_complete ==="
"$BIN" "$TMP/solid.yaml" "$TMP/out_solid" 2>&1 | grep -E "fill_complete|Unknown type" | head -3
exit 0
