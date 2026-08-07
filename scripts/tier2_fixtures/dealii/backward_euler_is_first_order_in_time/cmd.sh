#!/bin/bash
# Tier-2 for dealii navier_stokes#3 -- probe "time_integrator_order" of the shared
# translation unit _shared/ns_family.cc, compiled once and cached so every
# fixture naming a probe of that unit shares ONE C++ build. deal.II fixtures
# cost a C++ build, so the build count must not grow with the claim count.
#
# The 2D Taylor-Green vortex is an EXACT solution of the incompressible
# Navier-Stokes equations, so the time error is measured against the analytic
# answer with no reference run. Backward Euler against Crank-Nicolson.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" ns_family release time_integrator_order
