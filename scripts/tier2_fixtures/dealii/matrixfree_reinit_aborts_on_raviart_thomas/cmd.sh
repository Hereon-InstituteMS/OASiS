#!/bin/bash
# Tier-2 for dealii matrix_free#0 -- probes "mf_simplex_support" and
# "mf_rt_reinit" of the shared MatrixFree translation unit
# _shared/matrixfree_family.cc, compiled once and cached so four fixture
# directories share ONE C++ build.
#
# BOTH halves of the entry are executed. The out-of-date rule of thumb
# ("tensor-product elements only, not FE_SimplexP") is disproved on this build:
# MatrixFree<3>::reinit SUCCEEDS on FE_SimplexP(2) over a tetrahedral mesh from
# GridGenerator::subdivided_hyper_cube_with_simplices, and the cell_loop
# reproduces the assembled matrix-vector product to 2.9e-16.
#
# What does still fail is the vector-valued moment-based family, and it fails by
# ABORTING, so it runs in its own process here and the exit code is the
# observable: FE_RaviartThomas(0) gives rc=134 out of
# DEAL_II_NOT_IMPLEMENTED() inside
# internal::MatrixFreeFunctions::get_element_type_specific_information
# (matrix_free/shape_info.templates.h), printing "You are trying to use
# functionality in deal.II that is currently not implemented". The probe wraps
# the reinit in try/catch and prints caught_a_std_exception -- the line never
# appears, because an abort is not an exception. It is NOT gated on NDEBUG:
# rc=134 in the RELEASE build as well as the Debug one, which is why this
# fixture runs both.
#
# The grep checks the string the entry used to quote, "MatrixFree: element type
# not supported": it is in no header and no source file of this tree.
#
# Mutation control: T2_MUTATE=1 hands reinit an FE_Q(2) instead, both builds
# return rc=0, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

echo -n "phantom_element_type_message_files="
grep -rl "MatrixFree: element type not supported" /home/alexander/dealii/include \
  /home/alexander/dealii/source 2>/dev/null | wc -l

for variant in release debug; do
  echo "=== variant=$variant probe=mf_rt_reinit"
  out="$(bash "$SHARED/run.sh" matrixfree_family "$variant" mf_rt_reinit 2>&1 \
         | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_mf_rt_reinit_rc=${rc}"
done

echo "=== probe=mf_simplex_support"
exec bash "$SHARED/run.sh" matrixfree_family release mf_simplex_support
