#!/bin/bash
# Tier-2 for dealii contact#2 -- probe "contact_handmade_rows" of the shared
# contact translation unit _shared/contact_family.cc, compiled once and cached so
# three fixture directories share ONE C++ build.
#
# One CONVERGED active set from the iterated loop, imposed two ways on the same
# matrix, so the only thing that varies is how the constraint is applied: the
# hand-written route zeroes each constrained ROW and puts 1 on the diagonal, and
# AffineConstraints eliminates the constrained COLUMNS into the right-hand side
# as well.
#
# The entry calls the hand-written route "brittle". Measured, brittle means this:
# the operator stops being symmetric (relative symmetry defect 0.163), and CG --
# the solver this SPD problem calls for -- runs its whole 3000-iteration budget
# and returns with a residual of 113. This is one of the LOUD failures: a direct
# solve of the same unsymmetric system returns the AffineConstraints answer to
# 2e-16, so the constraint itself was imposed correctly. What the missing column
# elimination costs is the property the solver needs, not the answer.
#
# The entry's parallel-assembly claim is NOT tested: this deal.II is built without
# MPI.
#
# Mutation control: T2_MUTATE=1 eliminates the columns too, the operator is
# symmetric to machine precision, CG converges, and the fixture fails its own
# expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" contact_family release contact_handmade_rows
