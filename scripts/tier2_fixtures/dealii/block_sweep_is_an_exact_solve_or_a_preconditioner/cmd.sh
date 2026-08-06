#!/bin/bash
# Tier-2 for dealii dg_transport#3 -- probe "block_preconditioner_ratio" of the shared DG-transport
# translation unit _shared/dgtransport_family.cc, compiled once and cached so
# eleven fixture directories share ONE C++ build.
#
# PreconditionBlockSSOR with block_size = fe.n_dofs_per_cell() on the same
# upwind-DG operator in four regimes, each measured by applying the
# preconditioner ONCE to the rhs and reporting ||b - A P(b)|| / ||b||:
#   beta=(1,1) with the default cell order -- machine precision, an exact
#     block-triangular solve, GMRES needs one step while point Jacobi needs
#     hundreds, so no ratio between them is a property of the preconditioner;
#   beta=(1,-1) with the default order   -- an O(1) residual, an ordinary
#     preconditioner;
#   rotation about a CORNER, three levels -- O(1) residual and a block-SSOR
#     iteration count that GROWS with the mesh, while point SSOR stays within a
#     small factor of it and only point Jacobi scales badly;
#   rotation about the MIDDLE (closed characteristics) -- point Jacobi and point
#     SSOR both exhaust a 5000-iteration budget.
#
# T2_MUTATE=1 applies DoFRenumbering::downstream, and every regime except the
# closed-characteristic one collapses to an exact solve.
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dgtransport_family release block_preconditioner_ratio
