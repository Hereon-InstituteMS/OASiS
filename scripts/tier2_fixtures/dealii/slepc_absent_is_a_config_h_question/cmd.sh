#!/bin/bash
# Tier-2 for dealii eigenvalue#0: how to tell whether SLEPc is available, and
# what the compiler actually says when it is not.
#
# The claim's point is that "header not found" is the WRONG availability test.
# On a SOURCE build configured without SLEPc — the case here — the header IS
# installed and includes cleanly, because the whole file body sits behind
# #ifdef DEAL_II_WITH_SLEPC. The failure only surfaces when a class is named.
# The reliable test is config.h.
#
# Mutation control: T2_MUTATE=1 compiles the same file without naming any SLEPc
# class, so it builds and the diagnostic disappears.
set -u
REL=/home/alexander/dealii/build
SRC=/home/alexander/dealii
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT

echo "=== config.h is the availability test"
grep -E "DEAL_II_WITH_SLEPC|DEAL_II_WITH_PETSC" "$REL/include/deal.II/base/config.h" \
  | sed 's/^/config_h_/' || echo "config_h_no_slepc_line=true"
grep -q "^#define DEAL_II_WITH_SLEPC" "$REL/include/deal.II/base/config.h" \
  && echo "slepc_enabled=true" || echo "slepc_enabled=false"

echo "=== is the header present at all?"
if [ -f "$SRC/include/deal.II/lac/slepc_solver.h" ]; then
  echo "header_present=true"
else
  echo "header_present=false"
fi

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
printf '#include <deal.II/lac/slepc_solver.h>\nint main(){return 0;}\n' > "$W/inc.cc"
# shellcheck disable=SC2086
if /usr/bin/c++ $INC -fsyntax-only "$W/inc.cc" > "$W/inc.log" 2>&1; then
  echo "header_includes_cleanly=true"
else
  echo "header_includes_cleanly=false"
  grep -m2 -E "error" "$W/inc.log" | sed 's/^/include_error_/'
fi

echo "=== naming a SLEPc class"
if [ "${T2_MUTATE:-0}" = "1" ]; then
  printf '#include <deal.II/lac/slepc_solver.h>\nint main(){return 0;}\n' \
    > "$W/use.cc"
  echo "names_a_slepc_class=false"
else
  printf '#include <deal.II/lac/slepc_solver.h>\nint main(){ dealii::SLEPcWrappers::SolverKrylovSchur *p = nullptr; (void)p; return 0;}\n' \
    > "$W/use.cc"
  echo "names_a_slepc_class=true"
fi
# shellcheck disable=SC2086
if /usr/bin/c++ $INC -fsyntax-only "$W/use.cc" > "$W/use.log" 2>&1; then
  echo "compiles=true"
else
  echo "compiles=false"
  grep -m2 -E "error" "$W/use.log" | sed 's/^/compile_error_/'
fi
