#!/bin/bash
# Tier-2 for dealii dg_transport#4 -- probe "dof_wise_renumbering" of the shared DG
# translation unit _shared/dg_family.cc, compiled once per build type and cached
# so every fixture that names a probe of that unit shares ONE C++ build.
#
# DoFRenumbering::downstream(dof, direction, dof_wise_renumbering) on an upwind
# FE_DGQ(1) transport operator, then PreconditionBlockSSOR with
# block_size = fe.dofs_per_cell. The entry's premise is measured directly: with
# dof_wise_renumbering=TRUE not one of the 256 cells has contiguous dof indices
# any more, so the "blocks" the preconditioner inverts are not cells; with FALSE
# all 256 are contiguous.
#
# The consequence is measured by the entry's own diagnostic, one application of
# the preconditioner to the rhs:
#   dof_wise=true   ||b - A*P(b)||/||b|| = 1.07e+27
#   dof_wise=false  ||b - A*P(b)||/||b|| = 2.5e-15  (an exact block solve)
# Nothing is raised for it. The claim's "silent in every build" is why this
# fixture runs BOTH build types: the Debug library reproduces the release numbers
# digit for digit and exits 0, so there is no Assert to switch on here.
#
# Two clauses of the entry do NOT reproduce on this problem and the fixture pins
# them as measured: the residual is astronomical but NOT NaN, and GMRES does not
# report NoConvergence at step 0-1 with a NaN value -- it converges, in 4 steps
# with last_value 0 against 2 steps for the cell-wise numbering. The broken
# preconditioner is a silently wrong preconditioner, not a solver failure.
#
# Mutation control: T2_MUTATE=1 makes the probe pass dof_wise_renumbering=false,
# and the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

for variant in release debug; do
  echo "=== variant=$variant"
  out="$(bash "$SHARED/run.sh" dg_family "$variant" dof_wise_renumbering 2>&1)"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_rc=${rc}"
  if printf '%s\n' "$out" | grep -q \
      "^one_application_residual_is_nan_or_astronomical=true$"; then
    echo "summary_${variant}_block_preconditioner_is_broken=true"
  else
    echo "summary_${variant}_block_preconditioner_is_broken=false"
  fi
done
