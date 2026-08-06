#!/bin/bash
# Tier-2 for dealii matrix_free#4 -- probe "mf_cg_needs_a_preconditioner" of the
# shared multigrid translation unit _shared/multigrid_family.cc, compiled once
# and cached so five fixture directories share ONE C++ build.
#
# A MatrixFree FE_Q(2) Laplace operator at four refinement levels. At each level
# the matrix-free cell_loop is checked against the assembled SparseMatrix product
# on the same vector (relative difference ~3.6e-16), so the two iteration counts
# below are counts for the SAME operator and are comparable.
#
#   unpreconditioned CG   29, 60, 118, 238   -- doubling per refinement, the
#                                               O(h^-1) growth the entry names
#   GMG-preconditioned CG  6,  7,   7,   7   -- flat, and well inside the entry's
#                                               "~10-20"
#
# WHAT WAS NOT USED: the transfer here is MGTransferPrebuilt over assembled level
# matrices, not step-37's MGTransferMatrixFree. The claim under test is about the
# ITERATION COUNT, which is a property of the operator and the preconditioner and
# not of how the operator is applied; the matrix-free/assembled agreement above is
# what licenses reading it that way.
#
# Mutation control: T2_MUTATE=1 puts the GMG-preconditioned count under test, it
# stops growing, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" multigrid_family release mf_cg_needs_a_preconditioner
