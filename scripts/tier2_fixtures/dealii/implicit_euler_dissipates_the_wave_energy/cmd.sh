#!/bin/bash
# Tier-2 for dealii time_dependent_wave#0 -- probe "implicit_euler_energy" of the
# shared transient translation unit _shared/transient_family.cc, compiled once
# and cached so both fixtures of that unit share ONE C++ build.
#
# u_tt = laplace(u) on the unit square with u = 0 on the boundary, started from a
# smooth bump at rest, integrated 2000 steps at dt = 5e-3. The total energy
# 0.5 v^T M v + 0.5 c^2 u^T K u is a quadratic form in the SAME matrices the
# scheme uses, so it is exact; it is also computed the way the entry names it,
# from VectorTools::integrate_difference (L2 norm of u_t plus H1 seminorm of u),
# and the two agree to eight digits, which the run prints.
#
# Backward Euler on the first-order system loses 63% of the energy over that
# interval, and the loss is monotone at every one of the 2000 steps -- the entry's
# "monotonically DECAYING total energy" reproduces exactly.
#
# T2_MUTATE=1 switches to Newmark with beta = 1/4 and gamma = 1/2, whose energy
# ratio stays 1 with a relative drift of 4.4e-16 -- "conserve energy to roundoff"
# is the right description of it, and the fixture then fails its expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" transient_family release implicit_euler_energy
