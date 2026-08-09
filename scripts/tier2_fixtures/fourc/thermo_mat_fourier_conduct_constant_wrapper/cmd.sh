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
# Resolve the 4C binary: explicit override first, then the paths this
# repo has been verified against. Updated 2026-08-03 — the previous
# hard-coded $HOME/Schreibtisch/4C-src path no longer exists on the
# verification host; the deployed build is /home/alexander/4C/build/4C.
for _c in "${FOURC_BINARY:-}" "$HOME/4C/build/4C" "$HOME/Schreibtisch/4C-src/4C/build/4C"; do
  [ -x "$_c" ] && BIN="$_c" && break
done
: "${BIN:?4C binary not found — set FOURC_BINARY}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/4C-dependencies/lib}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# MUTATION CONTROL.  T2_MUTATE=1 removes the pathology: CONDUCT is written with
# the 'constant:' list wrapper the spec requires, MAT_Fourier matches, and the
# MATERIALS rejection never appears — so the fixture must go red.
MUTATE="${T2_MUTATE:-0}"
COND="      CONDUCT: 1.0"
[ "$MUTATE" = "1" ] && COND="      CONDUCT:
        constant: [1.0]"

cat > "$TMP/probe.yaml" <<EOF
PROBLEM TYPE:
  PROBLEMTYPE: Thermo
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
$COND
EOF
stdbuf -oL -eL "$BIN" "$TMP/probe.yaml" "$TMP/out" 2>&1 | head -25
exit 0
