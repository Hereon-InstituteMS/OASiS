#!/bin/bash
# Tier-2 for dealii multigrid#3 -- probe "mg_coarse_solver" of the shared
# multigrid translation unit _shared/multigrid_family.cc, compiled once and
# cached so five fixture directories share ONE C++ build.
#
# The FINE level is held fixed at 16641 dofs and the COARSEST level of the
# V-cycle is walked up (9, 25, 81, 289 dofs), which is exactly the experiment the
# entry's signal describes -- "an iteration count that grows with the COARSE-level
# DoF count while the fine level is unchanged". Both coarse solvers run at every
# setting, in the same program:
#   relaxation sweep as the coarse solve   7, 8, 11, 18 CG iterations
#   dense direct (MGCoarseGridHouseholder) 7, 7, 7, 7
#
# Mutation control: T2_MUTATE=1 puts the direct coarse solve under test, the count
# stops growing, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" multigrid_family release mg_coarse_solver
