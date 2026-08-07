#!/bin/bash
# Tier-2 for dealii wave#4 — probe "reflecting_boundary" of the shared wave
# translation unit _shared/wave_family.cc, compiled once and cached so six
# fixtures share one build.
#
# A Gaussian pulse on the unit square with the natural (do-nothing) boundary,
# integrated for two domain crossings: the discrete energy
# 0.5 v^T M v + 0.5 c^2 u^T K u is conserved to the digits printed, and the
# amplitude falls and then rises again — that rise is the reflected pulse.
#
# The absorbing form is the same code with a boundary damping term
# c * du/dt assembled on every boundary face (a boundary mass matrix entering
# the Newmark effective matrix as gamma*dt*C); it lets the energy leave.
#
# Mutation control: T2_MUTATE=1 turns that damping term on, and the fixture
# then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" wave_family release reflecting_boundary
