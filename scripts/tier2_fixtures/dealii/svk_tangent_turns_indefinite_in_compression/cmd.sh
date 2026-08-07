#!/bin/bash
# Tier-2 for dealii hyperelasticity#5 -- probe "svk_compression" of the shared
# translation unit _shared/hyperelastic_family.cc, compiled once and cached so
# every fixture that names a probe of that unit shares ONE C++ build.
#
# A 1x1 block (4x4 Q1, roller ends) is walked down a displacement ladder in 5%
# steps to 60% compression, once with S = lambda tr(E) I + 2 mu E
# (Saint-Venant-Kirchhoff) and once with the compressible Neo-Hookean law. After
# every converged step the smallest eigenvalue of the tangent restricted to the
# free dofs is computed with LAPACK, which is the observable that actually
# reports the loss of stability.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hyperelastic_family release svk_compression
