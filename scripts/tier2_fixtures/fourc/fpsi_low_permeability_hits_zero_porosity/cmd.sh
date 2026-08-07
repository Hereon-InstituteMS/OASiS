#!/bin/bash
# Tier-2 for fourc::fpsi#4 — permeability does control the difficulty, but the
# two ends fail in ways the entry did not predict.
#
# Claimed:  "stagnating CG/GMRES at very low K; for very high K, compare against
#            full NS to verify the Darcy regime applies".
# Observed, on upstream fpsi_ofsiinterface.4C.yaml (MAT_FluidPoro PERMEABILITY 1,
# UMFPACK on every field, two result tests):
#   1e-15 : no solver stagnation of any kind — a 4C abort, "zero porosity!",
#           from fluid_ele/4C_fluid_ele_calc_poro.cpp line 5397, raised inside
#           FluidEleCalcPoro::compute_stabilization_parameters.  The permeability
#           enters the porous stabilisation parameter, and the porosity check
#           there fires first.
#   1e+06 : runs to the end and returns a different answer, caught only by the
#           deck's own result tests.  Nothing suggests the Darcy assumption has
#           been left behind.
. "$(dirname "$0")/../_lib/preamble.sh"

BASE=$(upstream fpsi_ofsiinterface.4C.yaml) || exit 3
grep -q '      PERMEABILITY: 1' "$BASE" \
  || { echo "FIXTURE_ABORT=upstream_permeability_changed"; exit 3; }

# The two pathologies.
LOW_PERMEABILITY=1e-15
HIGH_PERMEABILITY=1e+06

cp "$BASE" "$TMP/nominal.yaml"
sed "s/      PERMEABILITY: 1$/      PERMEABILITY: $LOW_PERMEABILITY/"  "$BASE" > "$TMP/low.yaml"
sed "s/      PERMEABILITY: 1$/      PERMEABILITY: $HIGH_PERMEABILITY/" "$BASE" > "$TMP/high.yaml"
grep -m1 '      PERMEABILITY:' "$TMP/low.yaml"  | tr -d ' ' | sed 's/^/LOW_ARM_[/;s/$/]/'
grep -m1 '      PERMEABILITY:' "$TMP/high.yaml" | tr -d ' ' | sed 's/^/HIGH_ARM_[/;s/$/]/'

probe NOMINAL "$TMP/nominal.yaml"
probe LOW     "$TMP/low.yaml"
probe HIGH    "$TMP/high.yaml"

grep -m1 -F "OK (2)" "$TMP/NOMINAL.log"
grep -m1 -F "processor 0 finished normally" "$TMP/NOMINAL.log"
grep -m1 -F "zero porosity!" "$TMP/LOW.log"
grep -m1 -F "4C_fluid_ele_calc_poro.cpp" "$TMP/LOW.log"
grep -m1 -oF "compute_stabilization_parameters" "$TMP/LOW.log"
grep -m1 -F "Result check failed with 2 errors out of 2 tests" "$TMP/HIGH.log"
grep -m1 -F "is WRONG --> actresult=" "$TMP/HIGH.log"

# There is no stagnating iterative solver: every field is on UMFPACK.
echo "LOW_SOLVER_STAGNATION_MESSAGES=$(grep -ciE 'stagnat|maximum number of iterations|gmres|conjugate gradient' "$TMP/LOW.log")"
echo "LOW_REACHED_RESULT_TEST=$(grep -c 'Checking results of' "$TMP/LOW.log")"
# And the high-permeability end says nothing about leaving the Darcy regime.
echo "HIGH_FAILED_TESTS=$(grep -c 'is WRONG' "$TMP/HIGH.log")"
echo "HIGH_DARCY_WARNINGS=$(grep -ciE 'darcy|reynolds.*pore|permeability.*(large|invalid|range)' "$TMP/HIGH.log")"
exit 0
