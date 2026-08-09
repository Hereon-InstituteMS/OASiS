#!/bin/bash
# Tier-2 for dealii convection_diffusion#4 -- probe "anisotropic_layer" of the shared translation unit
# _shared/convdiff_family.cc, compiled once and cached so every fixture that names a
# probe of that unit shares ONE C++ build.
#
# Both strategies are run in every invocation and the whole refinement history is printed. What is measured is the nodal INTERPOLATION error of an exponential layer of width 1e-3, so the comparison is about how many cells each strategy needs to represent the layer with no solver in the way. cell->set_refine_flag(RefinementCase<2>::cut_axis(0)) is the anisotropic call.
#
# Mutation control: T2_MUTATE=1 makes the probe run the CORRECT variant, and the
# fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" convdiff_family release anisotropic_layer
