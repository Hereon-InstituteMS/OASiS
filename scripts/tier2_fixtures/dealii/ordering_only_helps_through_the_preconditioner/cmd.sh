#!/bin/bash
# Tier-2 for dealii advection_dg#3 -- probe "renumbering_and_gmres" of the shared DG-transport
# translation unit _shared/dgtransport_family.cc, compiled once and cached so
# eleven fixture directories share ONE C++ build.
#
# GMRES on the same DG operator under three dof orderings -- deal.II default,
# DoFRenumbering::Cuthill_McKee and DoFRenumbering::downstream(beta) -- with and
# without PreconditionBlockSSOR.
#
# UNPRECONDITIONED, the three counts come out IDENTICAL, which they must: a
# symmetric permutation of A and b permutes the whole Krylov space and cannot
# change an iteration count. The entry attributes the speed-up to the ordering
# alone ("~50-100 iters" down to "~10-20"); what the ordering actually changes is
# the PRECONDITIONER, whose sweep follows the numbering.
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dgtransport_family release renumbering_and_gmres
