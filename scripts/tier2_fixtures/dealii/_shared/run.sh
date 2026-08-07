#!/bin/bash
# usage: run.sh <name> <release|debug> [args...]
# Builds (cached) and runs one shared translation unit, echoing the deal.II
# variant and the process exit code so a fixture can assert on an ABORT.
# Assert() aborts with rc=134 rather than throwing, so the exit code IS the
# observable and a try/catch would see nothing.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="$1"; VARIANT="$2"; shift 2
EXE="$("$HERE/build.sh" "$NAME" "$VARIANT")" || { echo "BUILD_FAILED"; exit 1; }
echo "dealii_variant=$VARIANT"
timeout "${T2_TIMEOUT:-300}" "$EXE" "$@" 2>&1
echo "exit_code=$?"
