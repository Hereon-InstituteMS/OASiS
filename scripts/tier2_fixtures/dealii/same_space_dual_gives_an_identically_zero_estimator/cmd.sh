#!/bin/bash
# Tier-2 for dealii error_estimation#1 -- probe "dwr_same_space" of the shared
# goal-oriented translation unit _shared/goal_family.cc, compiled once and cached
# so three fixture directories share ONE C++ build.
#
# The same L-shaped goal-oriented problem, with the DUAL solved in the same Q1
# space as the primal. The weight of the dual-weighted residual is z - I_h z, and
# when z already lives in the primal space I_h z = z, so the weight is zero
# pointwise and the estimator is zero to machine precision -- at every cycle, and
# the mesh never refines either, because a criterion of all zeros selects no
# cells: the dof count stays at 225 for three cycles running.
#
# The entry says "the effectivity index is trivially 1". It is not: with a zero
# estimator over a nonzero error the effectivity index is 0. The index near 1 is
# what the HIGHER-ORDER dual delivers -- 0.94 in the mutated run, where the
# estimator tracks the goal error down three refinement cycles.
#
# T2_MUTATE=1 solves the dual in Q2, the estimator becomes nonzero and the
# fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" goal_family release dwr_same_space
