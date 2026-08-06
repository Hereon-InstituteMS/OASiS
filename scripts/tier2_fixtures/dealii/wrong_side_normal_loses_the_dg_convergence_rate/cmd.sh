#!/bin/bash
# Tier-2 for dealii advection_dg#2 -- probe "flipped_face_sides" of the shared DG-transport
# translation unit _shared/dgtransport_family.cc, compiled once and cached so
# eleven fixture directories share ONE C++ build.
#
# The interior-face term with the normal taken from the OTHER side while the jump
# [[v]] is left alone -- the sign error the entry describes -- against the correct
# assembly, at three refinement levels with a smooth manufactured solution.
#
# The entry says the result "is the TRANSPOSE of what was intended". It is not:
# the relative distance between the flipped matrix and the transpose of the
# correct one is printed and is O(1). And the consequence is worse than the
# claimed "degrades to O(1)": the L2 error GROWS under refinement, from ~2e4 to
# ~9e5, while the correct operator holds its second order.
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" dgtransport_family release flipped_face_sides
