#!/bin/bash
# Tier-2 for dealii hp_adaptive#3 — probe "no_smoothness_estimator" of the
# shared hp translation unit _shared/hp_family.cc, compiled once and cached so
# seven fixtures share one build.
#
# Eight adaptive cycles on a SMOOTH manufactured solution
# (u = sin(pi x) sin(pi y), zero Dirichlet), run twice in one process: once with
# the FESeries::Legendre object driving p-adaptivity, once without it. Without
# it there is no p-decision to make, every cell stays at FE_Q(1), and the
# L2 error against the analytic solution falls with the plain h-rate.
#
# The entry names a function "p_adaptivity_from_smoothness" that silently falls
# back to uniform refinement. No such symbol exists in this library — the grep
# below lists what hp::Refinement actually offers — so the pitfall is the
# absence of the estimator, not a fallback inside it.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== hp::Refinement p-adaptivity symbols that exist in this library"
grep -o "p_adaptivity_[a-z_]*" /home/alexander/dealii/include/deal.II/hp/refinement.h \
  | sort -u | sed 's/^/hp_refinement_symbol=/'
echo -n "p_adaptivity_from_smoothness_occurrences_in_headers="
grep -rc "p_adaptivity_from_smoothness" \
  /home/alexander/dealii/include/deal.II/hp/refinement.h || true

exec bash "$HERE/../_shared/run.sh" hp_family release no_smoothness_estimator
