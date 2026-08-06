#!/bin/bash
# Tier-2 for dealii hyperelasticity#2 -- probe "lnj_vs_squared_j" of the shared
# translation unit _shared/hyperelastic_family.cc, compiled once and cached so
# every fixture that names a probe of that unit shares ONE C++ build.
#
# Confined compression of a Neo-Hookean block (lateral faces held, so det F is
# exactly the axial stretch and the volumetric part of the law is what carries
# the load). The exact solution is a homogeneous deformation, which Q1
# reproduces exactly, so the discretisation error is zero by construction and
# any gap between S = mu(I-Cinv) + lambda*ln(J)*Cinv and the squared-J variant
# S = mu(I-Cinv) + lambda/2*(J^2-1)*Cinv is the MODEL difference alone. The same
# comparison is repeated on 4x4, 8x8 and 16x16, and the stress each law asks for
# is printed at det F = 0.5, 0.2 and 0.05.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hyperelastic_family release lnj_vs_squared_j
