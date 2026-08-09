#!/bin/bash
# Tier-2 for dealii obstacle_problem#0 -- probe "obstacle_active_set" of the
# shared goal-oriented translation unit _shared/goal_family.cc, compiled once and
# cached so three fixture directories share ONE C++ build.
#
# The step-41 membrane: -laplace(u) = -10 on (-1,1)^2 pushed down onto the
# staircase obstacle, u = 0 on the boundary, u >= psi. The Laplace matrix, its
# right-hand side and a lumped mass diagonal are assembled ONCE; only the
# AffineConstraints move between outer iterations, so what is compared is purely
# the active-set rule.
#
# The naive rule -- constrain the dofs that currently violate the obstacle -- is
# the one the entry describes, and it cycles with period two, measured rather
# than assumed: the loop records every active set it has seen and reports the
# period at which one repeats. 202 dofs, then 0, then 202 again. A dof pinned to
# the obstacle has zero gap, so the gap test releases it, and it violates again
# the moment it is free. The worst obstacle violation is back at 2.5 when the
# cycle closes.
#
# T2_MUTATE=1 switches to step-41's own rule, lambda + c * m_i * gap > 0, which
# also weighs the contact force. Two consecutive active sets become identical
# after four outer iterations -- inside the entry's stated 3-10 -- and the
# obstacle is satisfied everywhere. The fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" goal_family release obstacle_active_set
