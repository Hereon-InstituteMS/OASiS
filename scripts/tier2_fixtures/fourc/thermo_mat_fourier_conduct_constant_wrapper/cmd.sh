#!/bin/bash
# Tier-2: MAT_Fourier.CONDUCT requires a 'constant:' wrapper
# (list-typed), not a bare scalar.
#
# Inside MATERIALS, MAT_Fourier.CONDUCT is a tensor-valued
# property — even for isotropic conductivity the value must
# be wrapped as 'constant: [k]'. A bare scalar
# 'CONDUCT: 1.0' fails to match the MAT_Fourier input spec
# and 4C reports the whole MAT_Fourier block as 'remains
# unused'.
set -u
BIN=$HOME/Schreibtisch/4C-src/4C/build/4C
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/probe.yaml" <<'EOF'
PROBLEM TYPE:
  PROBLEMTYPE: Thermo
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT: 1.0
EOF
"$BIN" "$TMP/probe.yaml" "$TMP/out" 2>&1 | head -25
exit 0
