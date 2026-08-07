#!/bin/bash
# Tier-2 for dealii error_estimation#0 -- probe "dwr_primal_only" of the shared
# goal-oriented translation unit _shared/goal_family.cc, compiled once and cached
# so three fixture directories share ONE C++ build.
#
# -laplace(u) = 1 on the L-shaped domain with the r^(2/3) corner singularity at
# the origin, and a goal functional that is the mean of u over a square in the
# upper-left arm, nowhere near that corner. Three adaptive loops of five cycles
# run in the SAME program: one driven by the criterion under test, one by the
# dual-weighted residual, one by KellyErrorEstimator. The reference value of the
# functional comes from uniform Q2 solves at two refinement levels and the
# difference between them is printed, so the reference carries its own error bar.
#
# The entry's own number does not hold. It says the effectivity index is
# "typically O(1) to 10x off without the dual". Measured against the true goal
# error, the Kelly total is 253x off, while the dual-weighted estimator lands at
# 1.02 -- two orders of magnitude apart, not one.
#
# And "refines uniformly toward singularities regardless of the goal functional"
# is not what separates them: BOTH loops refine the reentrant corner, and the DWR
# loop actually places a slightly larger fraction of its cells there, because the
# dual solution is singular at that corner too. What the dual buys is the goal
# region, where it puts 23% of the cells against Kelly's 9%.
#
# T2_MUTATE=1 puts the dual-weighted criterion under test, and the fixture then
# fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" goal_family release dwr_primal_only
