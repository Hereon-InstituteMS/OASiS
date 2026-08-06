#!/bin/bash
# Tier-2 for dealii matrix_free#3 -- probe "mf_no_global_matrix" of the shared
# MatrixFree translation unit _shared/matrixfree_family.cc, compiled once and
# cached so four fixture directories share ONE C++ build.
#
# The entry states the observable as a PROFILE ("zero time in SparseMatrix::add",
# "the bulk of wall-clock in cell_loop"). Wall-clock is not a safe thing to assert
# on a shared machine, so what this fixture pins is STRUCTURAL and the timings are
# printed for information only: on a 4913-dof 3D FE_Q(2) problem the assembled
# path allocates a 1343472-byte SparseMatrix over a 167913-entry sparsity pattern,
# and the matrix-free path allocates neither -- MatrixFree stores no sparsity
# pattern at all. The two operators are checked against each other on the same
# vector and agree to 4.5e-16, so "no matrix" is not bought with a different
# answer.
#
# Mutation control: T2_MUTATE=1 puts the matrix-free path under test, no global
# matrix is allocated, and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" matrixfree_family release mf_no_global_matrix
