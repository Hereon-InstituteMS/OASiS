#!/bin/bash
# Tier-2 for dealii matrix_free#1 -- probes "mf_template_variants" and
# "mf_bad_template" of the shared MatrixFree translation unit
# _shared/matrixfree_family.cc, compiled once and cached so four fixture
# directories share ONE C++ build.
#
# ONE MatrixFree object (FE_Q(2) with QGauss<1>(3), 3D) and six FEEvaluation
# instantiations over it, each compared against the assembled SparseMatrix
# product on the same vector:
#   <3,2,3,1,double>  matching      4.1e-16
#   <3,-1,0,1,double> runtime form  4.1e-16, bit-for-bit the matching one
#   <3,3,3,1,double>  degree high   4.1e+12
#   <3,1,3,1,double>  degree low    3.1
#   <3,2,4,1,double>  n_q too high  5.5e+02
#   <3,2,2,1,double>  n_q too low   9.9e-01
# All six "ran without error" in Release -- nothing raised, nothing logged.
#
# One correction to the entry: it says the too-high degree produced NaN. It does
# not. On this build every one of the four mismatches returned a FINITE number,
# which is worse, because a NaN at least propagates visibly.
#
# The Debug half runs the degree-and-quadrature mismatch in its own process,
# where Assert fires from FEEvaluation::check_template_arguments and ABORTS
# (rc=134) with "Illegal arguments in constructor/wrong template arguments!".
# That is why this fixture sets requires_debug.
#
# Mutation control: T2_MUTATE=1 exercises only the matching and runtime forms,
# and the Debug probe uses the matching template, so both halves come back clean
# and the fixture fails its own expectations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED="$HERE/../_shared"

echo "=== variant=debug probe=mf_bad_template"
out="$(bash "$SHARED/run.sh" matrixfree_family debug mf_bad_template 2>&1 \
       | grep -vE '^(/media/|/lib/|/usr/lib|\[0x|#[0-9])')"
echo "$out"
rc="$(printf '%s\n' "$out" | sed -n 's/^exit_code=//p' | tail -1)"
echo "summary_debug_mf_bad_template_rc=${rc}"

echo "=== variant=release probe=mf_template_variants"
exec bash "$SHARED/run.sh" matrixfree_family release mf_template_variants
