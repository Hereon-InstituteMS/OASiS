#!/bin/bash
# Tier-2 for dealii mixed_laplacian#1 -- probe "cg_on_mixed_saddle_point" of the
# shared mixed-Laplacian translation unit _shared/mixed_family.cc, compiled once
# and cached so both fixtures of this topic share ONE C++ build.
#
# The step-20 system on an 8x8 mesh, FESystem(FE_RaviartThomas(0), FE_DGQ(0)):
# symmetric, with a zero pressure block. HOW indefinite it is gets counted rather
# than asserted -- the dense eigenvalues are computed and the negative ones come
# out at exactly the number of pressure dofs.
#
# Three solvers on the same matrix, in one run:
#   SolverCG on the full system     breaks down at step 1 with a NaN residual and
#                                   throws SolverControl::NoConvergence;
#   SolverMinRes on the full system converges without any preconditioner;
#   SolverCG on the Schur complement B^T M^{-1} B converges, and its pressure is
#                                   checked against a SparseDirectUMFPACK solve of
#                                   the whole system, so "converged" is not taken
#                                   on trust.
# The source is deliberately not a discrete eigenmode of the uniform-grid
# operator; with a sine source the Schur CG converges in a single step and the
# iteration count would say nothing.
#
# Mutation control: T2_MUTATE=1 puts the Schur-complement CG under test instead,
# it converges, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" mixed_family release cg_on_mixed_saddle_point
