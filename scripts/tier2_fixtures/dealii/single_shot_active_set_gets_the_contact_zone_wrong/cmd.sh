#!/bin/bash
# Tier-2 for dealii contact#0 -- probe "contact_single_shot" of the shared contact
# translation unit _shared/contact_family.cc, compiled once and cached so three
# fixture directories share ONE C++ build.
#
# A membrane on (-1,1)^2 pressed onto a rigid paraboloid indenter of finite
# extent, with the load chosen so the contact zone is a disc strictly INSIDE the
# indenter -- otherwise the contact radius is pinned by the geometry and measures
# nothing. Both strategies run in the same program: the active set predicted once
# from the unconstrained solve, and step-41's iterated loop.
#
# The entry's own diagnostic does NOT separate them. The outer contact radius is
# identical to eight digits (0.89976125 both ways), because the single-shot set
# reaches the rim of the indenter and so does the converged one. What is wrong is
# the AREA: 2601 active dofs against 184, an EQUIVALENT radius of 0.899 against
# 0.239 -- 276% out, not the "~30-50%" the entry quotes, and in a quantity the
# entry does not name.
#
# The iterated half reproduces exactly: two consecutive active sets agree after
# three outer iterations, inside the entry's "usually 3-10".
#
# Mutation control: T2_MUTATE=1 puts the iterated set under test and the fixture
# fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" contact_family release contact_single_shot
