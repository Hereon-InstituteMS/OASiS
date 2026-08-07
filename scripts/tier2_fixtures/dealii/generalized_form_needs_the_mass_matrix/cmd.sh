#!/bin/bash
# Tier-2 for dealii eigenvalue#3 — probe "standard_vs_generalized" of the shared eigenvalue translation unit
# _shared/eigen_family.cc (deflated inverse power iteration on K x = lambda M x,
# built-in SparseMatrix, because this deal.II has neither PETSc nor SLEPc).
# Compiled once and cached; six fixtures share the one build.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" eigen_family release standard_vs_generalized
