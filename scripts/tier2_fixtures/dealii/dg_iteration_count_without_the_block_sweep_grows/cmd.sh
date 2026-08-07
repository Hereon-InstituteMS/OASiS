#!/bin/bash
# Tier-2 for dealii dg_advection_reaction#1 -- probe "iteration_count_vs_h" of the shared DG-transport
# translation unit _shared/dgtransport_family.cc, compiled once and cached so
# eleven fixture directories share ONE C++ build.
#
# GMRES iteration counts on the DG-advection operator at four refinement levels
# for the curved field (rotation about a corner), with point Jacobi, with no
# preconditioner at all, and with PreconditionBlockSSOR over downstream-ordered
# cell blocks. SolverCG is tried once on the same non-symmetric operator and
# fails, so "a plain SolverCG" is not an option to compare against.
#
# NO MULTIGRID WAS BUILT. The entry credits geometric multigrid for
# h-independence; measured here, the block sweep over downstream-ordered cells
# alone already converges in one or two steps at every level, so multigrid is not
# what h-independence requires on this operator.
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dgtransport_family release iteration_count_vs_h
