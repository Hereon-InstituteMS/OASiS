#!/bin/bash
# Tier-2 for dealii convection_diffusion#0 -- probe "supg_tau" of the shared translation unit
# _shared/convdiff_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# 1D transport, b = 1, 20 cells, inflow 0 and outflow 1, run at three cell Peclet numbers (0.05, 0.5, 5). Plain Galerkin is solved alongside in the same run so the undershoot the claim records is visible in every run, and the tau under test is compared with the analytic solution AT THE NODES -- the doubly-asymptotic tau is nodally exact in 1D, the bare h/(2|b|) is not.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" convdiff_family release supg_tau
