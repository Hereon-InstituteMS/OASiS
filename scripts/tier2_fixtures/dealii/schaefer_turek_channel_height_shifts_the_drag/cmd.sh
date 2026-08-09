#!/bin/bash
# Tier-2 for dealii stokes#6 -- probe "benchmark_geometry" of the shared translation unit
# _shared/stokes_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# GridGenerator::channel_with_cylinder against GridGenerator::uniform_channel_with_cylinder, which fixes the cylinder diameter at one and requires the channel extents to be INTEGER multiples of it, so 2.2 x 0.41 with the cylinder at (0.2, 0.2) cannot be expressed and the closest it reaches is a symmetric 2.2 x 0.40 channel. Both grids come out with the same cell count at both levels, so the drag comparison is not a resolution artefact.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stokes_family release benchmark_geometry
