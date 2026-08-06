#!/bin/bash
# Tier-2 for dealii nonlinear#2 -- probe "stale_linearisation" of the shared translation unit
# _shared/nonlinear_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# The nonlinear coefficient is evaluated at the INITIAL guess for both the tangent and the residual and never updated -- the 'assemble_linearisation used a stale solution' mistake. A properly updated Newton run on the same mesh is computed in the same invocation and used as the reference, and the true nonlinear residual at the accepted answer is printed alongside the residual the frozen loop reports.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" nonlinear_family release stale_linearisation
