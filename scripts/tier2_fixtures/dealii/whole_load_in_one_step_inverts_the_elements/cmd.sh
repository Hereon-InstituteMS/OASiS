#!/bin/bash
# Tier-2 for dealii hyperelasticity#0 -- probe "load_stepping" of the shared
# translation unit _shared/hyperelastic_family.cc, compiled once and cached so
# every fixture that names a probe of that unit shares ONE C++ build.
#
# Total-Lagrangian compressible Neo-Hookean cantilever (12x3 Q1, 104 dofs),
# body force 400 in -y. The SAME total load is applied (a) all at once from the
# undeformed cold start and (b) in twenty increments, each starting from the
# previous converged state. Nothing in deal.II owns the Newton loop, so the
# residual history printed here is the probe's own.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hyperelastic_family release load_stepping
