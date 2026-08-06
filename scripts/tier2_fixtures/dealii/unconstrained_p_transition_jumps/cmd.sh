#!/bin/bash
# Tier-2 for dealii hp_adaptive#4 — probe "p_interface_without_constraints" of
# the shared hp translation unit _shared/hp_family.cc, compiled once and cached
# so seven fixtures share one build.
#
# A UNIFORM 4x4 mesh — no h-hanging nodes anywhere — with FE_Q(1) on the left
# half and FE_Q(3) on the right. Every constraint that
# DoFTools::make_hanging_node_constraints produces on this mesh is therefore the
# p-projection the entry is about (the probe prints how many there are).
#
# Solved without them, the trace of the solution differs on the two sides of
# each p-transition face: the probe walks every interior face, evaluates both
# neighbours with FEFaceValues at matched physical points, and reports the worst
# difference. Faces between equal-degree cells are the control and stay
# continuous to round-off, so the jump is specific to the p-transition.
#
# Mutation control: T2_MUTATE=1 applies the constraints, and the fixture then
# fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hp_family release p_interface_without_constraints
