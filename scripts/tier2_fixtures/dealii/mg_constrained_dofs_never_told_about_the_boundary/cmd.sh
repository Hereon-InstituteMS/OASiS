#!/bin/bash
# Tier-2 for dealii multigrid#1 -- probe "mg_boundary_indices" of the shared
# multigrid translation unit _shared/multigrid_family.cc, compiled once and
# cached so five fixture directories share ONE C++ build.
#
# A step-16 FE_Q(2) Laplace V-cycle over five levels with ONLY
# MGConstrainedDoFs::make_zero_boundary_constraints omitted.
# have_boundary_indices() comes back false and get_boundary_indices(level) is
# empty on every one of the five levels, printed level by level with the level
# dof counts beside them so "on EVERY level, not just the finest" is read off the
# run rather than assumed. The V-cycle still runs -- it is not a valid
# preconditioner, but it does not raise -- and CG needs 145 iterations where the
# correctly constrained hierarchy needs single digits.
#
# The adaptive half of the entry is checked in the same run on a second mesh with
# one corner refined: get_refinement_edge_indices(level) is EMPTY on the four
# fully refined levels and NON-EMPTY (17 indices) exactly on the level carrying
# the refinement edge, which is what the entry says to look for.
#
# Mutation control: T2_MUTATE=1 calls make_zero_boundary_constraints, every level
# carries its own boundary index set, and the fixture fails its expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" multigrid_family release mg_boundary_indices
