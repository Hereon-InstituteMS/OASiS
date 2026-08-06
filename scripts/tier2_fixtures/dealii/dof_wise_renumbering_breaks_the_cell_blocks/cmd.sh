#!/bin/bash
# Tier-2 for dealii dg_transport#4 -- probe "dof_wise_renumbering" of the shared
# DG-transport translation unit _shared/dgtransport_family.cc, compiled once and
# cached so eleven fixture directories share ONE C++ build.
#
# DoFRenumbering::downstream(dof, direction, dof_wise_renumbering) with the third
# argument TRUE. Whether a cell's dofs are still contiguous is COUNTED, not
# assumed: all 256 cells lose contiguity, and the one-application residual of
# PreconditionBlockSSOR over "blocks" of fe.n_dofs_per_cell() comes back at
# ~9e+26 instead of the machine-precision exact solve the cell-wise numbering
# gives.
#
# The entry says this is "silent in every build", so the fixture RUNS BOTH
# LIBRARIES and pins the pair: rc=0 with identical output from the Release and
# the Debug deal.II, no Assert anywhere. That is why it sets requires_debug.
#
# The other half of the entry does NOT reproduce: GMRES did not report
# NoConvergence with a NaN value at step 0-1. It converged, and
# SolverControl::last_value() was finite and zero.
#
# Mutation control: T2_MUTATE=1 passes dof_wise_renumbering=false, the blocks are
# cells again, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

for variant in release debug; do
  echo "=== variant=$variant"
  out="$(bash "$SHARED/run.sh" dgtransport_family "$variant" dof_wise_renumbering 2>&1 \
         | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
  echo "$out"
  rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
  echo "summary_${variant}_rc=${rc}"
done
