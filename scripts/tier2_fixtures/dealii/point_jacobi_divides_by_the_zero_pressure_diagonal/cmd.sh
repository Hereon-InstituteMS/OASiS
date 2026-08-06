#!/bin/bash
# Tier-2 for dealii stokes#3 -- probe "jacobi_vs_vanka" of the shared translation unit
# _shared/stokes_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# The relaxation the claim names is tested directly as the GMRES preconditioner on the assembled Stokes matrix rather than inside a full geometric-multigrid V-cycle: PreconditionJacobi is the very object MGSmootherRelaxation would hold. T2_MUTATE=1 swaps in SparseVanka with the pressure dofs selected, which is the Vanka-type smoother of step-56.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stokes_family release jacobi_vs_vanka
