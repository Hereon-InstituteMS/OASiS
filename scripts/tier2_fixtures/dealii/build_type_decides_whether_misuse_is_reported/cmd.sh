#!/bin/bash
# Tier-2 for dealii poisson#7: the SAME misuse produces a full diagnostic on a
# Debug build and silence or a segfault on a Release build.
#
# Three verified pairs, run here against both libraries on this host:
#   SparseMatrix::add() outside the sparsity pattern  Debug abort / Release drop
#   active_fe_index beyond the hp::FECollection       Debug abort / Release SEGV
#   get_function_gradients, scalar container on a     Debug abort / Release
#     vector-valued FESystem                            returns a wrong mixture
#
# Assert ABORTS (rc=134); it does not throw, so the exit code is the observable
# and a try/catch would see nothing. rc=139 is the segfault.
#
# Mutation control: T2_MUTATE=1 makes every probe use the CORRECT index / shape,
# so no build aborts and the Debug/Release split disappears.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

echo "=== build type of each library on this host"
grep -h "^CMAKE_BUILD_TYPE" /home/alexander/dealii/build/CMakeCache.txt \
  | sed 's/^/release_tree_/'
grep -h "^CMAKE_BUILD_TYPE" \
  "${DEAL_II_DEBUG_DIR:-/media/alexander/PortableSSD/dealii-verify-r2/dbgbuild}/CMakeCache.txt" \
  | sed 's/^/debug_tree_/'
ls /home/alexander/dealii/build/lib/libdeal_II.so >/dev/null 2>&1 \
  && echo "release_library_present=true" || echo "release_library_present=false"
ls "${DEAL_II_DEBUG_DIR:-/media/alexander/PortableSSD/dealii-verify-r2/dbgbuild}/lib/libdeal_II.g.so" \
  >/dev/null 2>&1 \
  && echo "debug_library_present=true" || echo "debug_library_present=false"

for probe in sparse_add_outside_pattern hp_active_fe_index \
             vector_valued_gradients; do
  for variant in release debug; do
    echo "=== probe=$probe variant=$variant"
    out="$(bash "$SHARED/run.sh" assert_family "$variant" "$probe" 2>&1)"
    echo "$out"
    rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
    echo "summary_${probe}_${variant}_rc=${rc}"
  done
done
