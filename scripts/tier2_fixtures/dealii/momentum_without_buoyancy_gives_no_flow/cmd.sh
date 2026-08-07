#!/bin/bash
# Tier-2 for dealii time_dependent_ns#1 -- probe "buoyancy_coupling" of the shared
# transient translation unit _shared/transient_family.cc, compiled once and cached
# so both fixtures of that unit share ONE C++ build.
#
# The canonical Boussinesq test the entry names: a side-heated cavity, hot left
# wall and cold right wall, no-slip everywhere, in the infinite-Prandtl (Stokes)
# form that step-31 uses. Taylor-Hood Q2/Q1 for velocity and pressure, Q2 for the
# temperature, both DoFHandlers on one triangulation, and ten coupled steps so a
# flow that only appears through the coupling has every chance to appear.
#
# With the momentum equation solved "in isolation" -- no Ra * T * e_y term -- the
# right-hand side of the Stokes system is the zero vector and the velocity comes
# back IDENTICALLY zero: max speed 0, kinetic energy 0, after the first solve and
# again after ten coupled steps. The temperature field is a perfectly good
# nonzero field the whole time; it simply never reaches the momentum equation.
#
# T2_MUTATE=1 puts the buoyancy term back at Ra = 1e4: max speed 39 and kinetic
# energy 335 on the first solve. The fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" transient_family release buoyancy_coupling
