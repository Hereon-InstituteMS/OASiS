#!/bin/bash
# Tier-2 for dealii phase_field#1 -- probe "fixed_supg_tau" of the shared
# stabilisation translation unit _shared/stabilisation_family.cc, compiled once
# and cached so three fixture directories share ONE C++ build.
#
# The same 1D layer on 20 cells, at cell Peclet 0.05 and 0.5, with three schemes
# in the same run: plain Galerkin, the fixed tau = h/(2|b|), and the
# doubly-asymptotic tau = h/(2|b|) * (coth(Pe) - 1/Pe).
#
# In the diffusion-dominated regime the fixed tau is measurably WORSE than doing
# nothing: its worst nodal error is about fifty times the plain-Galerkin one,
# while the doubly-asymptotic tau is nodally exact to machine precision at both
# Peclet numbers.
#
# The entry's two numbers do not both hold. The gradient smearing, measured as
# 1 - (discrete slope of the last element)/(exact slope), reaches the quoted ~20%
# at cell Peclet 0.5 but is only about 3% at cell Peclet 0.05, so "~20% even in
# clearly resolved regions" overstates the resolved end. And the entry's
# diagnostic points the wrong way: the KellyErrorEstimator total of the
# over-stabilised solution is SMALLER than the nodally exact one at both Peclet
# numbers, because an over-diffused solution has smoother gradients across faces.
# Kelly rewards the error it is supposed to expose.
#
# Mutation control: T2_MUTATE=1 puts the doubly-asymptotic tau under test, and the
# fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" stabilisation_family release fixed_supg_tau
