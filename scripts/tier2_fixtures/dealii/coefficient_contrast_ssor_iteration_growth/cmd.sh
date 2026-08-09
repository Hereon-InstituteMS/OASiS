#!/bin/bash
# Tier-2 for dealii poisson#6 — probe "coefficient_contrast" of the shared adaptive-Poisson
# translation unit _shared/poisson_family.cc, compiled once and cached.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" poisson_family release coefficient_contrast
