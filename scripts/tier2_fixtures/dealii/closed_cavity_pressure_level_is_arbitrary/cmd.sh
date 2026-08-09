#!/bin/bash
# Tier-2 for dealii stokes#2 -- probe "pressure_nullspace" of the shared translation unit
# _shared/stokes_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# Lid-driven cavity, velocity Dirichlet on the whole boundary. The probe checks that the constant-pressure vector is in the kernel of the assembled matrix, then runs the SAME unrestarted GMRES twice differing only in the starting vector: the velocity agrees to 1e-10 and the pressure shape agrees to 1e-10, but the pressure LEVEL differs by exactly the shift put into the start.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stokes_family release pressure_nullspace
