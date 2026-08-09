#!/bin/bash
# Tier-2 for fourc::porous_media#2 — PERMEABILITY = 0 is fatal, but WHICH
# material you set it on decides whether 4C tells you so.  Neither of the two
# diagnostics the entry quoted (`zero pivot detected`, `solver returned status:
# -3`) exists in 4C or in any Trilinos library it links.
#
#   MAT_FluidPoro (Poroelasticity, poro_2D_quad4_linporo)
#       -> named guard: "zero or negative permeability", mat/4C_mat_fluidporo.cpp
#
#   MAT_FluidPoroMultiPhase (porofluid_pressure_based_2D_quad4)
#       -> NO guard.  The flux-reconstruction solve at t=0 returns an all-zero
#          residual, the first nonlinear step blows up from 5.99e-01 to 5.13e+14,
#          and the process dies on SIGFPE inside the element evaluator
#          (EvaluatorMassPressure::evaluate_matrix_and_assemble), exit 136.
#
# That asymmetry is the useful thing: an agent that sets zero permeability on the
# multiphase material gets a floating-point crash with no message naming the
# input it got wrong.
. "$(dirname "$0")/../_lib/preamble.sh"

PORO=$(upstream poro_2D_quad4_linporo.4C.yaml) || exit 3
MULTI=$(upstream porofluid_pressure_based_2D_quad4.4C.yaml) || exit 3
ln -s "$(dirname "$MULTI")/xml" "$TMP/xml"
grep -q "PERMEABILITY: 0.01" "$PORO"  || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }
grep -q "PERMEABILITY: 1"    "$MULTI" || { echo "FIXTURE_ABORT=upstream_deck_changed"; exit 3; }

cp "$PORO" "$TMP/poro_ok.yaml"
sed 's/      PERMEABILITY: 0.01/      PERMEABILITY: 0.0/' "$PORO"  > "$TMP/poro_zero.yaml"
sed 's/^      PERMEABILITY: 1$/      PERMEABILITY: 0/'    "$MULTI" > "$TMP/multi_zero.yaml"

probe POROOK    "$TMP/poro_ok.yaml"
probe POROZERO  "$TMP/poro_zero.yaml"
probe MULTIZERO "$TMP/multi_zero.yaml"

grep -m1 -F "processor 0 finished normally" "$TMP/POROOK.log"
# MAT_FluidPoro: 4C names the mistake.
grep -m1 -F "zero or negative permeability" "$TMP/POROZERO.log"
grep -m1 -oF "4C_mat_fluidporo.cpp" "$TMP/POROZERO.log"
# MAT_FluidPoroMultiPhase: no guard, a floating-point crash instead.
grep -m1 -F "Signal: Floating point exception (8)" "$TMP/MULTIZERO.log"
grep -m1 -oF "EvaluatorMassPressure" "$TMP/MULTIZERO.log"
# 4C's own error banner ("PROC 0 ERROR in <file>, line <n>:") is what a reader
# greps for.  The MAT_FluidPoro arm emits one; the multiphase arm emits none at
# all and just dies on the signal.
echo "POROZERO_HAS_4C_ERROR_BANNER=$(grep -c 'PROC 0 ERROR in' "$TMP/POROZERO.log")"
echo "MULTIZERO_HAS_4C_ERROR_BANNER=$(grep -c 'PROC 0 ERROR in' "$TMP/MULTIZERO.log")"
# The residual explodes rather than the solver reporting anything.
echo "MULTIPHASE_RESIDUAL_BLOWUP=$(grep -c '5.128e+14' "$TMP/MULTIZERO.log")"
# Neither quoted diagnostic exists.
echo "CLAIMED_ZERO_PIVOT_TEXT=$(grep -ci 'zero pivot' "$TMP/POROZERO.log" "$TMP/MULTIZERO.log" | awk -F: '{s+=$2} END {print s}')"
echo "CLAIMED_SOLVER_STATUS_MINUS3_TEXT=$(grep -ci 'solver returned status' "$TMP/POROZERO.log" "$TMP/MULTIZERO.log" | awk -F: '{s+=$2} END {print s}')"
exit 0
