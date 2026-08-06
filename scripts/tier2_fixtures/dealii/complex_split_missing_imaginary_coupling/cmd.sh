#!/bin/bash
# Tier-2 for dealii helmholtz#1 — probe "complex_split_coupling" of the shared
# translation unit _shared/helmholtz_family.cc. This deal.II is a real-scalar
# build, so the complex Helmholtz problem is split into a 2-component FESystem
# carrying (u_re, u_im); the absorbing boundary term -i k u v dS is purely
# off-diagonal in that split, and omitting it leaves the two blocks independent.
#
# Mutation control: T2_MUTATE=1 assembles the coupling, and the fixture then
# fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" helmholtz_family release complex_split_coupling
