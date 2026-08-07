#!/bin/bash
# Tier-2 for dealii nonlinear#1 -- probe "line_search" of the shared translation unit
# _shared/nonlinear_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# Same minimal surface problem at boundary amplitude 4, and the SAME initial guess in both variants (zero interior with the boundary values interpolated), so the only difference is whether the step length is backtracked. The full-step run prints its residual history and the inner SolverCG failure text, which is the only library-side message in the whole loop -- deal.II has no Newton solver.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" nonlinear_family release line_search
