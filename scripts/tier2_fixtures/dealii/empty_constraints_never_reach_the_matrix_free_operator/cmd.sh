#!/bin/bash
# Tier-2 for dealii matrix_free#2 -- probe "mf_missing_constraints" of the shared
# MatrixFree translation unit _shared/matrixfree_family.cc, compiled once and
# cached so four fixture directories share ONE C++ build.
#
# A 3D FE_Q(2) Laplace problem with 386 real Dirichlet constraints, with ONLY the
# AffineConstraints object handed to MatrixFree::reinit changed. Handed an empty
# one, reinit returns normally and raises nothing, and
# MatrixFree::get_constrained_dofs().size() comes back 0 -- the cheap check the
# entry recommends, read straight after reinit.
#
# The downstream consequence is measured in the same run: CG on the matrix-free
# operator runs to its 300-iteration limit with the residual blown up to 1.6e+16,
# and the "solution" reaches 7.4e+30 on a boundary where it should be zero.
#
# Mutation control: T2_MUTATE=1 hands reinit the real constraints,
# get_constrained_dofs().size() is nonzero, CG converges to a solution that is
# zero on the boundary, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" matrixfree_family release mf_missing_constraints
