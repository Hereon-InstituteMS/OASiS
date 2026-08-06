#!/bin/bash
# Tier-2 for dealii wave#0 — probe "newmark_beta_stability" of the shared wave
# translation unit _shared/wave_family.cc (Newmark in acceleration form on
# d^2u/dt^2 = c^2 laplacian(u), built-in SparseMatrix + CG/SSOR). Compiled once
# and cached; six fixtures share the one build.
#
# beta = 0.10 < gamma/2 = 0.25 at dt = 0.05: the amplitude leaves the bounded
# range within 20 steps. The SAME dt with the canonical (0.25, 0.5) pair is run
# in the same process as the contrast and stays at O(0.2).
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant
# (beta = 0.25), and the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" wave_family release newmark_beta_stability
