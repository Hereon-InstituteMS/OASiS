#!/bin/bash
# Tier-2 for dealii time_dependent_heat#0 -- probe "no_solution_transfer" of the
# shared transient-heat translation unit _shared/heat_family.cc, which is
# compiled once and cached, so this fixture adds no build of its own.
#
# Three backward-Euler steps on a 16x16 mesh, then every cell flagged for
# refinement and execute_coarsening_and_refinement() called. Without
# SolutionTransfer the state does NOT survive: the new DoFHandler has a different
# number of dofs and the reinit'ed vector is exactly zero, so the l2 norm drops
# from O(1) to 0. It is the ZERO-VECTOR branch of the entry's Signal that
# happens, not the "random noise" one -- the old vector is not reinterpreted, it
# is replaced.
#
# T2_MUTATE=1 runs the canonical step-26 sequence
# (prepare_for_coarsening_and_refinement -> refine -> interpolate) and the state
# is carried across, so the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" heat_family release no_solution_transfer
