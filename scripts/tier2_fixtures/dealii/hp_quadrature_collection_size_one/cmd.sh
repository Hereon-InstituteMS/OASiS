#!/bin/bash
# Tier-2 for dealii hp_adaptive#2 — probe "single_quadrature_rule" of the shared
# hp translation unit _shared/hp_family.cc, compiled once and cached so seven
# fixtures share one build.
#
# An hp mesh mixing FE_Q(1..4) with a variable coefficient, assembled three
# times in one process: with the matched hp::QCollection, with a single
# QGauss(4), and with a single QGauss(2). A collection of size 1 is BROADCAST to
# every element (the probe reproduces exactly what hp::FEValues does with a
# one-entry collection), which is the usual shape of this bug.
#
# The entry's two "widespread myths" are re-tested rather than repeated:
#   symmetry — max|A_ij - A_ji| is printed for the matched and the coarse
#     assembly; both stay at round-off, so symmetry is NOT the tell;
#   the graded damage — QGauss(2) leaves the operator singular and SolverCG
#     runs to its limit and throws SolverControl::NoConvergence with a residual
#     that has grown by orders of magnitude, while QGauss(4) converges normally
#     and only shifts a rule-independent functional (the integral of the
#     solution, always evaluated with the matched collection).
#
# Mutation control: T2_MUTATE=1 makes the run under test use the matched
# collection, and the fixture then fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/../_shared/run.sh" hp_family release single_quadrature_rule
