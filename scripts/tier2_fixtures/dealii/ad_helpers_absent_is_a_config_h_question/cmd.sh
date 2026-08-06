#!/bin/bash
# Tier-2 for dealii nonlinear#3: the Differentiation::AD helpers the claim tells
# you to choose between do not exist on every install, and "the header is there"
# is the WRONG availability test.
#
# The whole body of deal.II/differentiation/ad/ad_helpers.h -- CellLevelBase,
# EnergyFunctional, ResidualLinearization -- sits behind
#   #if defined(DEAL_II_WITH_ADOLC) || defined(DEAL_II_TRILINOS_WITH_SACADO)
# so on a source build configured without either (the case here) the header is
# installed, includes cleanly, and yields NOTHING. The failure only surfaces
# when a class is named. The reliable test is config.h.
#
# The claim's substantive assertion -- that swapping EnergyFunctional for
# CellLevelBase gives a tangent with the wrong sign on the off-diagonals -- is
# NOT testable on this install, because neither AD backend is present.
#
# Mutation control: T2_MUTATE=1 compiles the same file without naming any AD
# class, so it builds and the diagnostic disappears.
set -u
REL=/home/alexander/dealii/build
SRC=/home/alexander/dealii
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT

echo "=== config.h is the availability test"
grep -E "DEAL_II_WITH_ADOLC|DEAL_II_TRILINOS_WITH_SACADO|DEAL_II_WITH_TRILINOS" \
  "$REL/include/deal.II/base/config.h" | sed 's/^/config_h_/'
grep -q "^#define DEAL_II_WITH_ADOLC" "$REL/include/deal.II/base/config.h" \
  && echo "adolc_enabled=true" || echo "adolc_enabled=false"
grep -q "^#define DEAL_II_TRILINOS_WITH_SACADO" \
  "$REL/include/deal.II/base/config.h" \
  && echo "sacado_enabled=true" || echo "sacado_enabled=false"
if grep -q "^#define DEAL_II_WITH_ADOLC" "$REL/include/deal.II/base/config.h" \
   || grep -q "^#define DEAL_II_TRILINOS_WITH_SACADO" \
        "$REL/include/deal.II/base/config.h"; then
  echo "ad_enabled=true"
else
  echo "ad_enabled=false"
fi

echo "=== is the header present at all?"
if [ -f "$SRC/include/deal.II/differentiation/ad/ad_helpers.h" ]; then
  echo "header_present=true"
else
  echo "header_present=false"
fi
echo "=== what guards the class definitions"
sed -n '18p' "$SRC/include/deal.II/differentiation/ad/ad_helpers.h" \
  | sed 's/^/guard_line_/'

INC="-isystem $REL/include -isystem $SRC/include \
 -isystem $SRC/bundled/taskflow-3.10.0 \
 -isystem $SRC/bundled/boost-1.84.0/include \
 -isystem $SRC/bundled/kokkos-4.5.01/algorithms/src \
 -isystem $SRC/bundled/kokkos-4.5.01/containers/src \
 -isystem $SRC/bundled/kokkos-4.5.01/core/src \
 -isystem $SRC/bundled/kokkos-4.5.01/simd/src \
 -isystem $SRC/bundled/kokkos-4.5.01/tpls/mdspan/include \
 -isystem $SRC/bundled/kokkos-4.5.01/tpls/desul/include \
 -isystem $SRC/bundled/magic_enum-v0.9.7/include/magic_enum"

echo "=== does the header INCLUDE cleanly?"
printf '#include <deal.II/differentiation/ad.h>\nint main(){return 0;}\n' \
  > "$W/inc.cc"
# shellcheck disable=SC2086
if /usr/bin/c++ $INC -fsyntax-only "$W/inc.cc" > "$W/inc.log" 2>&1; then
  echo "header_includes_cleanly=true"
else
  echo "header_includes_cleanly=false"
  grep -m2 -E "error" "$W/inc.log" | sed 's/^/include_error_/'
fi

echo "=== naming the two AD helpers the claim tells you to choose between"
if [ "${T2_MUTATE:-0}" = "1" ]; then
  printf '#include <deal.II/differentiation/ad.h>\nint main(){return 0;}\n' \
    > "$W/use.cc"
  echo "names_an_ad_class=false"
else
  {
    printf '#include <deal.II/differentiation/ad.h>\n'
    printf 'int main(){\n'
    printf '  dealii::Differentiation::AD::EnergyFunctional<\n'
    printf '    dealii::Differentiation::AD::NumberTypes::sacado_dfad, double>\n'
    printf '      *a = nullptr;\n'
    printf '  dealii::Differentiation::AD::CellLevelBase<\n'
    printf '    dealii::Differentiation::AD::NumberTypes::sacado_dfad, double>\n'
    printf '      *b = nullptr;\n'
    printf '  (void)a; (void)b; return 0;\n}\n'
  } > "$W/use.cc"
  echo "names_an_ad_class=true"
fi
# shellcheck disable=SC2086
if /usr/bin/c++ $INC -fsyntax-only "$W/use.cc" > "$W/use.log" 2>&1; then
  echo "compiles=true"
else
  echo "compiles=false"
  grep -m2 -E "error" "$W/use.log" | sed 's/^/compile_error_/'
fi
